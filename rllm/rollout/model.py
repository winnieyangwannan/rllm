
from vllm import SamplingParams
from rllm.system_prompts import COT_MATH_SYSTEM_PROMPT


class BaseRollout:
    def __init__(self, engine):
        self.engine = engine
    
    def rollout(self, **generation_kwargs):
        pass

    def generate(self, messages, **generation_kwargs):
        responses = self.engine.chat(messages, SamplingParams(**generation_kwargs))
        
        preds = []
        for resp in responses:
            if generation_kwargs.get('n', 1) > 1:
                pred = [output.text for output in resp.outputs]
            else:
                pred = resp.outputs[0].text
            preds.append(pred)

        return preds

        
class COTRollout(BaseRollout):
    
    def rollout(self, queries, **generation_kwargs):
        messages = [
            [
                {"role": "system", "content": COT_MATH_SYSTEM_PROMPT},
                {"role": "user", "content": queries},
            ]
            for query in queries
        ]

        return self.generate(messages, **generation_kwargs)