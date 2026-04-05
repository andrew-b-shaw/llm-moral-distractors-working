"""HuggingFace Qwen 3 VL (vision-language) model wrapper.

Handles interleaved image+text prompts using AutoModelForImageTextToText and
the qwen_vl_utils.process_vision_info helper for image preprocessing."""

import os
import torch
import math

from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.generation.utils import GenerateDecoderOnlyOutput

from src.config import PATH_HF_CACHE, PATH_OFFLOAD, PATH_DATA
from src.models.model_utils import get_timestamp, get_api_key
from src.models.model import LanguageModel, MODELS
from src.models.qwen_model import QwenModelResponse
from src.prompters.prompt import Modality, Prompt, ImagePosition


class QwenVLModel(LanguageModel):
    """Hugging Face Qwen 3 VL (vision-language) backend."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "QwenVLModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        login(token=get_api_key("huggingface"))
        if MODELS[model_name]["8bit"]:
            raise ValueError(f"Unknown Model '{model_name}'")
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype="auto",
                device_map="auto",
                cache_dir=PATH_HF_CACHE,
                offload_folder=PATH_OFFLOAD
            )

        self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE
        )

        self._device = next(self._model.parameters()).device

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> QwenModelResponse:
        image_inputs, video_inputs = None, None
        distractor = prompt["distractor"]
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt["system_prompt"]}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt["user_prompt"]}
                ]
            }
        ]

        if distractor and distractor["modality"] == Modality.IMAGE:
            image_path = f"{PATH_DATA}/{distractor["file_path"]}"
            image_message = {"type": "image", "image": image_path}
            match distractor["position"]:
                case ImagePosition.BEFORE_SYSTEM:
                    messages[0]["content"].insert(0, image_message)
                case ImagePosition.AFTER_SYSTEM:
                    messages[0]["content"].append(image_message)
                case ImagePosition.BEFORE_USER:
                    messages[1]["content"].insert(0, image_message)
                case _:
                    messages[1]["content"].append(image_message)
            image_inputs, video_inputs = process_vision_info(messages)

        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            response = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                length_penalty=0,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )

        answer_raw = self._processor.decode(response.sequences[0], skip_special_tokens=True).strip()
        answer = answer_raw[len(text_prompt):].strip()

        return QwenModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=response,
            tokenizer=self._tokenizer
        )