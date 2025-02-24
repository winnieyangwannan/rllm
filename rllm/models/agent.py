from abc import ABC, abstractmethod
import openai
import time


class BaseAgent(ABC):

    def __init__(
        self,
        rollout_engine,
        engine_name,
        tokenizer,
        api_key=None,
        api_retries=3,
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tokenizer = tokenizer
        self.engine_name = engine_name
        self.api_key = api_key
        self.api_retries = api_retries

        self.sampling_params = kwargs.get("sampling_params", None)

    @abstractmethod
    def _pre_get_action(self, obs):
        """
        Automatically called before get_action.
        Return in same format as OpenAI ChatInterface.
        """
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
        ]

    @abstractmethod
    def _post_get_action(self, response):
        """
        Automatically called after get_action.
        Return a string
        """
        return response

    @abstractmethod
    def update(self, action, observation, next_observation, reward, terminated, truncated, info):
        """
        Automatically called after taking an environment step during environment interation.
        No return required
        """
        return

    @abstractmethod
    def reset(self):
        return

    def get_action(self, obs,  **kwargs):
        """
        Ideally should NOT be modified or override
        """
        if self.engine_name == "verl":
            return self._get_action_verl(obs,  **kwargs)
        elif self.engine_name == "vllm":
            return self._get_action_vllm(obs,  **kwargs)
        elif self.engine_name == "openai":
            return self._get_action_openai(obs,  **kwargs)
        else:
            raise NotImplementedError

    def _get_action_verl(self, obs,  **kwargs):
        from verl.utils.model import compute_position_id_with_mask
        from verl import DataProto
        from verl.protocol import union_two_dict

        prompt = self._pre_get_action(obs,  **kwargs)
        prompts = [prompt]
        # because of veRL's chunking. we need to pad number of prompts to be a multiple of worker group world size
        padding_needed = (
            self.rollout_engine.world_size
            - len(prompts) % self.rollout_engine.world_size
        ) % self.rollout_engine.world_size

        if padding_needed > 0:
            prompts.extend([[{"role": "system", "content": ""}]] * padding_needed)

        inputs = self.tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)

        # original_batch contains the extra info needed for generation
        if "original_batch" in kwargs and kwargs["original_batch"]:
            original_batch = kwargs["original_batch"]
            # only use the original_batch's meta_info since tensor_batch is from batch_dict and non_tensor_batch is not neeeded
            data.meta_info = union_two_dict(data.meta_info, original_batch.meta_info)

        output = self.rollout_engine.generate_sequences(data)
        output_text = self.tokenizer.batch_decode(
            output.batch["responses"], skip_special_tokens=False
        )
        # remove the padding
        if padding_needed > 0:
            output_text = output_text[:-padding_needed]

        pad_token = self.tokenizer.pad_token

        responses = []
        for i, text in enumerate(output_text):
            rsp = text.replace(pad_token, "")
            responses.append(rsp)

        assert (
            len(responses) == 1
        ), f"Only 1 answer should be generated, but {len(responses)} is"
        response = responses[0]
        action = self._post_get_action(obs, prompt, response,  **kwargs)

        return action

    def _get_action_vllm(self, obs,  **kwargs):
        prompt = self._pre_get_action(obs,  **kwargs)

        prompts = [prompt]
        prompts = self.tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=False
        )
        # Generate responses using vLLM
        outputs = self.rollout_engine.generate(
            prompts=prompts, sampling_params=self.sampling_params, use_tqdm=False
        )

        # Decode the output token IDs into text
        responses = []
        # Get the generated text directly from the RequestOutput object
        for i, output in enumerate(outputs):
            rsp = output.outputs[0].text
            responses.append(rsp)

        assert (
            len(responses) == 1
        ), f"Only 1 answer should be generated, but {len(responses)} is"
        response = responses[0]
        action = self._post_get_action(obs, prompt, response,  **kwargs)

        return action

    def _get_action_openai(self, obs,  **kwargs):
        prompt = self._pre_get_action(obs,  **kwargs)

        openai.api_key = self.api_key

        retries = self.api_retries
        while retries > 0:
            try:
                # OpenAI API call
                openai_response = openai.chat.completions.create(
                    model="o1-preview", messages=prompt
                )
                response = openai_response.choices[0].message.content
            except openai.RateLimitError:
                retries -= 1
                if retries == 0:
                    return "Error: Rate limit reached and retries exhausted."
                print(f"Sleep for 5 seconds for API limit.")
                time.sleep(5)
            except Exception as e:
                return f"Error processing content: {e}"

        action = self._post_get_action(obs, prompt, response,  **kwargs)

        return action
