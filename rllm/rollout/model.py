
from vllm import SamplingParams


class vLLMModel:
    def __init__(self, vllm_instance):
        self.vllm = vllm_instance
    
    def __call__(self, **generation_kwargs):
        pass

    def rollout(self, messages, **generation_kwargs):
        responses = self.vllm.chat(messages, SamplingParams(**generation_kwargs))
        
        preds = []
        for resp in responses:
            if generation_kwargs.get('n', 1) > 1:
                pred = [output.text for output in resp.outputs]
            else:
                pred = resp.outputs[0].text
            preds.append(pred)

        return preds

        
class ProverRollout(vLLMModel):
    
    def __call__(self, questions, **generation_kwargs):
        messages = [
            [
                {"role": "system", "content": "You are an expert problem solver. You will be given a problem to solve. Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": f"{question}"},
            ] for question in questions
        ]

        return self.rollout(messages, **generation_kwargs)