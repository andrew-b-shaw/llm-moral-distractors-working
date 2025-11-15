from __future__ import annotations

import os
import torch
import math
from PIL import Image
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GemmaTokenizer, pipeline
from transformers.generation.utils import GenerateDecoderOnlyOutput

from src.config import PATH_HF_CACHE, PATH_OFFLOAD
from src.models.model_utils import get_timestamp, get_api_key
from src.models.models import LanguageModel, MODELS, LanguageModelResponse
from src.prompters.prompt import Distractor, Modality


class QwenModelResponse(LanguageModelResponse):
    _output: GenerateDecoderOnlyOutput
    _tokenizer: AutoTokenizer

    def __init__(
        self,
        timestamp: str,
        answer_raw: str,
        answer: str,
        output: GenerateDecoderOnlyOutput,
        tokenizer: AutoTokenizer
    ):
        super().__init__(
            timestamp=timestamp,
            answer_raw=answer_raw,
            answer=answer
        )
        self._output = output
        self._tokenizer = tokenizer

    def get_answer_prob(self, answer: str) -> float:
        """
        Returns probability that the output **starts** with given string

        :param answer: the string to calculate the probability of
        :return: the probability that the output **starts** with the given string
        """
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

class QwenModel(LanguageModel):
    """Gemma 3 Model Wrapper --> Supports text + image prompts"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "QwenModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        # Setup access using HF login
        login(token=get_api_key("huggingface"))

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE,
            device_map="auto",
            offload_folder=PATH_OFFLOAD,
            dtype=torch.bfloat16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        self._device = next(self._model.parameters()).device

    def query(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        distractor: Distractor | None = None
    ) -> QwenModelResponse:
        """
        Query Gemma model (with top-p decoding)

        :param system_prompt: the system prompt to query the model with
        :param user_prompt: the user prompt to query the model with
        :param max_tokens: the max output tokens
        :param temperature: the temperature to generate outputs with
        :param top_p: the probability to use for top_p decoding
        :param distractor: the distractor to inject (optional)
        :return: a GemmaModelResponse with the model output
        """
        if distractor is not None:
            if distractor["modality"] == Modality.IMAGE:
                raise Exception("This model does not support image inputs!")

            text_path = f"{os.path.abspath(os.getcwd())}/data/{distractor['file_path']}"
            with open(text_path, 'r') as f:
                distractor_text = f.read()
                user_prompt = f"{distractor_text} Later, {user_prompt}"

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self._tokenizer(
            [text],
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self._tokenizer.eos_token_id,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )

        answer_raw = self._tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        answer = answer_raw[len(text) - 1:].strip()

        return QwenModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=output,
            tokenizer=self._tokenizer
        )
