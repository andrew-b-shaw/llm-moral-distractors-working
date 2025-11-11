import math
from datetime import time

import openai
from pathlib import Path

from src.models.models import LanguageModel, MODELS, API_TIMEOUTS, LanguageModelResponse
from src.models.model_utils import get_timestamp, get_api_key


class OpenAIModelResponse(LanguageModelResponse):
    _output: any

    def __init__(
            self,
            timestamp: str,
            answer: str,
            answer_raw: str,
            output: any,
    ):
        super().__init__(timestamp, answer, answer_raw)

    def get_answer_prob(self, answer: str) -> float:
        token_ids = self._tokenizer(answer).input_ids
        if len(token_ids) - 1 > len(self._output.logits):
            return 0.0

        answer_log_prob = 0.0
        for i in range(len(token_ids) - 1):
            token_id = token_ids[i + 1]
            logits = self._output.logits[i]
            token_probs = torch.softmax(logits, dim=1).squeeze()
            answer_log_prob += math.log(token_probs[token_id].item())

        return math.exp(answer_log_prob)

class OpenAIModel(LanguageModel):
    """OpenAI API Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OpenAIModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        api_key = get_api_key("openai")
        # openai.api_key = api_key
        self._client = openai.OpenAI(api_key=api_key)

    def _prompt_request(
            self,
            prompt_base: str,
            prompt_system: str,
            max_tokens: int,
            temperature: float = 0.0,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            image_path: str = None,
    ):
        success = False
        t = 0

        while not success:
            try:
                # Dialogue Format
                user_content = [{"type": "text", "text": prompt_base}]
                if image_path:
                    # Append image to same user message
                    image_url = f"data:image/{Path(image_path).suffix[1:]};base64,{self._encode_image(image_path)}"
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})

                messages = [
                    {"role": "system", "content": f"{prompt_system[:-2]}"},
                    {"role": "user", "content": user_content},
                ]

                # Query ChatCompletion endpoint
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logprobs=True,
                    top_logprobs=20
                )

                # Set success flag
                success = True

            except Exception as e:
                print(e)
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS) - 1)

        return response

    def get_greedy_answer(
            self, prompt_base: str, prompt_system: str, max_tokens: int, image_path: str = None
    ) -> str:
        return self.query(
            user_prompt=prompt_base,
            system_prompt=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
            image_path=image_path
        )

    def query(
            self,
            user_prompt: str,
            system_prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            image_path: str = None,
    ) -> any:
        result = {
            "timestamp": get_timestamp(),
        }

        # (1) Top-P Sampling
        response = self._prompt_request(
            prompt_base=user_prompt,
            prompt_system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            image_path=image_path
        )

        completion = response.choices[0].message.content.strip()

        result["answer_raw"] = completion.strip()
        result["answer"] = completion.strip()

        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        token_probs = {
            "Yes": 0,
            "yes": 0,
            "No": 0,
            "no": 0,
            "A": 0,
            "a": 0,
            "B": 0,
            "b": 0,
            " Yes": 0,
            " No": 0,
            " yes": 0,
            " no": 0,
            " A": 0,
            " B": 0,
            " a": 0,
            " b": 0
        }
        for logprob in logprobs:
            if logprob.token in token_probs.keys():
                token_probs[logprob.token] = math.exp(logprob.logprob)

        result["token_prob_yes"] = token_probs["Yes"] + token_probs["yes"] + token_probs[" Yes"] + token_probs[" yes"]
        result["token_prob_no"] = token_probs["No"] + token_probs["no"] + token_probs[" No"] + token_probs[" no"]
        result["token_prob_a"] = token_probs["A"] + token_probs["a"] + token_probs[" A"] + token_probs[" a"]
        result["token_prob_b"] = token_probs["B"] + token_probs["b"] + token_probs[" B"] + token_probs[" b"]

        return result