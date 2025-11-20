import math
from datetime import time

import openai
from openai import OpenAI

from pathlib import Path
import os

import torch

from src.models.models import LanguageModel, MODELS, API_TIMEOUTS, LanguageModelResponse
from src.models.model_utils import get_timestamp, get_api_key
from src.prompters.prompt import Distractor, Modality
from transformers.generation.utils import GenerateDecoderOnlyOutput
import tiktoken

class OpenAIModelResponse(LanguageModelResponse):
    model="cl100k_base" # "gpt-4o"
    # _tokenizer = tiktoken.encoding_for_model(model)
    _tokenizer = tiktoken.get_encoding("cl100k_base") 

    _output: GenerateDecoderOnlyOutput

    def __init__(
            self,
            timestamp: str,
            answer: str,
            answer_raw: str,
            output: GenerateDecoderOnlyOutput,
    ):
        super().__init__(timestamp, answer, answer_raw)
        self._output = output
        # tokenizer = tiktoken.encoding_for_model("gpt-4o")
        tokenizer = tiktoken.get_encoding("cl100k_base") 
        self._tokenizer = tokenizer

    def get_answer_prob(self, answer: str) -> float:
        token_ids = self._tokenizer.encode(answer).input_ids
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
        self._client = OpenAI(api_key=api_key)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        
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
            self, prompt: str, max_tokens: int, image_path: str = None
    ) -> str:
        return self.query(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
            image_path=image_path
        )

    def query(
            self,
            prompt: dict,
            max_tokens: int,
            temperature: float = 0.0,
            top_p: float = 1.0,
            distractor: Distractor | None = None,
            enable_thinking: bool = False
    ) -> OpenAIModelResponse:
        """
        OpenAI API wrapper
        """

        distractor_obj = prompt.get("distractor")
        if distractor_obj is not None and distractor_obj["modality"] == Modality.IMAGE:
            raise ValueError("This model does not support image inputs!")

        # this was apply_chat_template before? 
        # I (catherine) changed it to format_messages to match other models
        def format_messages(messages, add_generation_prompt=True, enable_thinking=False):
            prompt_text = ""
            for m in messages:
                content = m["content"]
                if isinstance(content, list):  # user_content may be a list of dicts
                    for c in content:
                        if "text" in c:
                            prompt_text += c["text"] + "\n"
                else:
                    prompt_text += content + "\n"

            if add_generation_prompt:
                prompt_text += "\nAnswer:"
            return prompt_text

        messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_prompt"]}
        ]

        text_prompt = format_messages(messages, add_generation_prompt=True, enable_thinking=enable_thinking)

        # API call
        success = False
        t = 0
        while not success:
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": prompt["system_prompt"]},
                        {"role": "user", "content": prompt["user_prompt"]}
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    logprobs=True,
                )
                success = True
            except Exception as e:
                print(e)
                import time
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS) - 1)

        choice = response.choices[0]

        if choice.logprobs is not None and choice.logprobs.content is not None:
            for token_obj in choice.logprobs.content:
                print(f"Token: {token_obj.token!r}, LogProb: {token_obj.logprob:.6f}")
        else:
            print("No logprobs available for this model.")
                
        answer_raw = response.choices[0].message["content"]
        answer = answer_raw.strip()

        return OpenAIModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=response,        # API response here
            tokenizer=self._tokenizer
        )