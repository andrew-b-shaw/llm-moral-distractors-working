"""HuggingFace Llama 3.2 Vision-Instruct model wrapper.

Llama-3.2-11B/90B-Vision-Instruct are Mllama checkpoints, not registered under
AutoModelForCausalLM, so they need AutoModelForImageTextToText instead of the
plain text LlamaModel wrapper. Text-only prompts (no image distractor) still
generate normally since Mllama's cross-attention layers only engage when
pixel_values are provided."""

from __future__ import annotations

import math

import torch
from PIL import Image
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast
from transformers.generation.utils import GenerateDecoderOnlyOutput

from src.config import PATH_HF_CACHE, PATH_OFFLOAD, PATH_DISTRACTORS
from src.models.model_utils import get_timestamp, get_api_key
from src.models.model import LanguageModel, MODELS, LanguageModelResponse
from src.prompters.prompt import Modality, Prompt, ImagePosition


class LlamaVisionModelResponse(LanguageModelResponse):
    _output: GenerateDecoderOnlyOutput
    _tokenizer: PreTrainedTokenizerFast

    def __init__(
        self,
        timestamp: str,
        answer_raw: str,
        answer: str,
        output: GenerateDecoderOnlyOutput,
        tokenizer: PreTrainedTokenizerFast
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
        # Skip token_ids[0] (BOS token added by the tokenizer), matching LlamaModelResponse.
        token_ids = self._tokenizer(answer).input_ids
        if len(token_ids) - 1 > len(self._output.logits):
            return 0.0

        answer_log_prob = 0.0
        for i in range(len(token_ids) - 1):
            token_id = token_ids[i + 1]
            logits = self._output.logits[i]
            token_probs = torch.softmax(logits, dim=1).squeeze()
            token_prob = token_probs[token_id].item()
            if token_prob == 0.0:
                return 0.0
            answer_log_prob += math.log(token_prob)

        return math.exp(answer_log_prob)


class LlamaVisionModel(LanguageModel):
    """Llama 3.2 Vision-Instruct Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "LlamaVisionModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        # Setup access using HF login
        login(token=get_api_key("huggingface"))

        if MODELS[model_name]["8bit"]:
            raise ValueError(f"Unknown Model '{model_name}'")
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
                dtype=torch.bfloat16,
            )

        self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

        self._device = next(self._model.parameters()).device

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> LlamaVisionModelResponse:
        """
        Query Llama Vision model (with top-p decoding)

        :param prompt: the Prompt to query the model with
        :param max_tokens: the max output tokens
        :param temperature: the temperature to generate outputs with
        :param top_p: the probability to use for top_p decoding
        :return: a LlamaVisionModelResponse with the model output
        """
        distractor = prompt["distractor"]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt["system_prompt"]}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt["user_prompt"]}]
            }
        ]

        image = None
        if distractor and distractor["modality"] == Modality.IMAGE:
            image_message = {"type": "image"}
            match distractor["position"]:
                case ImagePosition.BEFORE_SYSTEM:
                    messages[0]["content"].insert(0, image_message)
                case ImagePosition.AFTER_SYSTEM:
                    messages[0]["content"].append(image_message)
                case ImagePosition.BEFORE_USER:
                    messages[1]["content"].insert(0, image_message)
                case _:
                    messages[1]["content"].append(image_message)
            image_path = f"{PATH_DISTRACTORS}/{distractor["file_path"]}"
            image = Image.open(image_path).convert("RGB")

        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )

        answer_raw = self._processor.decode(output.sequences[0], skip_special_tokens=True).strip()
        answer = answer_raw[len(text_prompt):].strip()

        return LlamaVisionModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=output,
            tokenizer=self._tokenizer
        )
