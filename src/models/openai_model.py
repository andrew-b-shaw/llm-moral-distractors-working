import math
import warnings

import tiktoken
import time
import json
import pandas as pd

from openai import OpenAI, ChatCompletion

from src.config import PATH_DATA
from src.models.models import LanguageModel, LanguageModelResponse
from src.models.model_configs import MODELS, API_TIMEOUTS
from src.models.model_utils import get_timestamp, get_api_key
from src.prompters.prompt import Modality, Prompt
from tiktoken import Encoding

class OpenAIModelResponse(LanguageModelResponse):
    _tokenizer: Encoding
    _top_logprobs: list[dict[str, float]]

    def __init__(
        self,
        timestamp: str,
        answer: str,
        answer_raw: str,
        output: ChatCompletion
    ):
        super().__init__(timestamp, answer, answer_raw)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._top_logprobs = []
        for token in output.choices[0].logprobs.content:
            self._top_logprobs.append(dict([
                (top_logprob.token, top_logprob.logprob)
                for top_logprob in token.top_logprobs
            ]))

    def get_answer_prob(self, answer: str) -> float:
        answer_tokens = [self._tokenizer.decode([token_id]) for token_id in self._tokenizer.encode(answer)]
        if len(answer_tokens) > len(self._top_logprobs):
            return 0.0

        answer_logprob = 0.0
        for answer_token, top_logprob in zip(answer_tokens, self._top_logprobs):
            if answer_token in top_logprob:
                answer_logprob += top_logprob[answer_token]
            else:
                return 0.0

        return math.exp(answer_logprob)

class OpenAIModel(LanguageModel):
    """OpenAI API Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OpenAIModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        api_key = get_api_key("openai")
        # openai.api_key = api_key
        self._client = OpenAI(api_key=api_key)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> OpenAIModelResponse:
        """
        OpenAI API wrapper
        """
        distractor_obj = prompt.get("distractor")
        if distractor_obj is not None and distractor_obj["modality"] == Modality.IMAGE:
            raise ValueError("This model does not support image inputs!")

        # API call
        messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_prompt"]}
        ]
        success = False
        t = 0
        while not success:
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=20
                )
                success = True
            except Exception as e:
                print(e)
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS) - 1)

        answer_raw = response.choices[0].message.content
        answer = answer_raw.strip()

        return OpenAIModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=response
        )


class OpenAIBatchSubmitResponse(LanguageModelResponse):
    def get_answer_prob(self, answer: str) -> float:
        return 0.0


class OpenAIBatchSubmitModel(LanguageModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        warnings.warn("Prompter post-processing may need to be manually disabled when submitting a batch request!")
        warnings.warn("Filename MUST be set manually in the OpenAIBatchSubmitModel config!")

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> LanguageModelResponse:
        with open(PATH_DATA / MODELS[self._model_name]["output_filepath"], 'a') as f:
            request = {
                "custom_id": prompt["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1",
                    "messages": [
                        {"role": "system", "content": prompt["system_prompt"]},
                        {"role": "user", "content": prompt["user_prompt"]}
                    ],
                    "max_completion_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "logprobs": True,
                    "top_logprobs": 20
                }
            }
            f.write(json.dumps(request))
            f.write("\n")

        return OpenAIBatchSubmitResponse(
            timestamp=get_timestamp(),
            answer_raw="",
            answer=""
        )


class OpenAIBatchRetrieveModelResponse(LanguageModelResponse):
    def get_answer_prob(self, answer: str) -> float:
        return 0.0


class OpenAIBatchRetrieveModel(LanguageModel):
    _responses: dict[str, str]

    def __init__(self, model_name: str):
        super().__init__(model_name)
        warnings.warn("Filename MUST be set manually in the OpenAIBatchSubmitModel config!")
        self._responses = {}
        responses_df = pd.read_csv(PATH_DATA / MODELS[self._model_name]["input_filepath"])
        for i, row in responses_df.iterrows():
            self._responses[row["prompt_id"]] = row["answer_raw"]

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> LanguageModelResponse:
        if prompt["id"] not in self._responses:
            raise ValueError("The given prompt ID is not in the returned responses for the batch!")

        answer_raw = self._responses[prompt["id"]]
        answer = answer_raw.strip()

        return OpenAIBatchRetrieveModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer
        )
