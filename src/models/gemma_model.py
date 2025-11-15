from __future__ import annotations

import torch
import math
from PIL import Image
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GemmaTokenizer, pipeline
from transformers.generation.utils import GenerateDecoderOnlyOutput

from src.config import PATH_HF_CACHE, PATH_OFFLOAD, PATH_DISTRACTORS
from src.models.model_utils import get_timestamp, get_api_key
from src.models.models import LanguageModel, MODELS, LanguageModelResponse
from src.prompters.prompt import Modality, Prompt, Position


class GemmaModelResponse(LanguageModelResponse):
    _output: GenerateDecoderOnlyOutput
    _tokenizer: GemmaTokenizer

    def __init__(
        self,
        timestamp: str,
        answer_raw: str,
        answer: str,
        output: GenerateDecoderOnlyOutput,
        tokenizer: GemmaTokenizer
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
        token_ids = self._tokenizer.encode(answer)
        if len(token_ids) > len(self._output.logits):
            return 0.0

        answer_log_prob = 0.0
        for i in range(len(token_ids) - 1):
            token_id = token_ids[i + 1]  # first token ID is BOS
            logits = self._output.logits[i]  # last token ID is EOS
            token_probs = torch.softmax(logits, dim=1).squeeze()
            answer_log_prob += math.log(token_probs[token_id].item())

        return math.exp(answer_log_prob)


class GemmaModel(LanguageModel):
    """Gemma 3 Model Wrapper"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "GemmaModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        # Setup access using HF login
        login(token=get_api_key("huggingface"))

        # Load model (may include vision)
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE,
            device_map="auto",
            offload_folder=PATH_OFFLOAD,
            dtype=torch.bfloat16,
        )
        self._device = next(self._model.parameters()).device

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        try:
            # Vision-capable Gemma models have a processor
            self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        except Exception as e:
            self._processor = None
            print(e)

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GemmaModelResponse:
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

        distractor = prompt["distractor"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt["system_prompt"]},  # Gemma doesn't support system prompts (https://ai.google.dev/gemma/docs/core/prompt-structure)
                    {"type": "text", "text": prompt["user_prompt"]}
                ]
            }
        ]

        if distractor and distractor["modality"] == Modality.IMAGE:
            if not self._processor:
                raise ValueError(
                    f"Model '{self._model_name}' does not support images. Try a vision-capable Gemma variant."
                )

            image_message = {"type": "image"}
            match distractor["position"]:
                case Position.BEFORE_SYSTEM:
                    messages[0]["content"].insert(0, image_message)
                case Position.AFTER_SYSTEM:
                    messages[0]["content"].insert(1, image_message)
                case Position.BEFORE_USER:
                    messages[0]["content"].insert(1, image_message)
                case _:
                    messages[0]["content"].append(image_message)

            text_prompt = self._processor.apply_chat_template(messages)
            image_path = f"{PATH_DISTRACTORS}/{distractor["file_path"]}"
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=[prompt],
                images=[image],
                return_tensors="pt"
            ).to(self._device)
        else:
            text_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            inputs = self._tokenizer(
                text=text_prompt,
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
        answer = answer_raw[answer_raw.rfind("model") + len("model"):].strip()

        return GemmaModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=output,
            tokenizer=self._tokenizer
        )