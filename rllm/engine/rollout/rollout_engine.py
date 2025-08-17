from dataclasses import dataclass


@dataclass
class ModelOutput:
    text: str
    tool_calls: list
    finish_reason: str
    completion_tokens: int
    prompt_tokens: int


class RolloutEngine:
    def __init__(self, model: str, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError("get_model_response is not implemented")

    def wake_up(self):
        pass

    def sleep(self):
        pass
