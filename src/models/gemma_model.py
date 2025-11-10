import torch
from PIL import Image
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GemmaTokenizer

from src.config import PATH_HF_CACHE, PATH_OFFLOAD
from src.models.model_utils import get_timestamp, get_api_key
from src.models.models import LanguageModel, MODELS, LanguageModelResponse
from src.prompters.distractor import Distractor


class GemmaModelResponse(LanguageModelResponse):
    _output: any
    _tokenizer: GemmaTokenizer

    def __init__(
        self,
        timestamp: str,
        answer: str,
        answer_raw: str,
        output: any,
        tokenizer: GemmaTokenizer
    ):
        super().__init__(timestamp, answer, answer_raw)
        self._output = output
        self._tokenizer = tokenizer

    def get_answer_prob(self, answer: str) -> float:
        token_ids = self._tokenizer(answer).input_ids
        if len(token_ids) - 1 >= len(self._output.scores):
            return 0.0

        answer_prob = 1.0
        for i in range(len(token_ids) - 1):
            token_id = token_ids[i + 1]
            logits = self._output.scores[i]
            token_probs = torch.softmax(logits, dim=1).squeeze()
            answer_prob *= token_probs[token_id]

        return answer_prob

class GemmaModel(LanguageModel):
    """Gemma 2 Model Wrapper --> Supports text + image prompts"""

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

        # Tokenizer + (optional) Processor
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)

        try:
            # Vision-capable Gemma models have a processor
            self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE, )
        except Exception as e:
            self._processor = None
            print(e)

        self._device = next(self._model.parameters()).device

    def generate_with_image(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GemmaModelResponse:
        """
        Generate a response from Gemma with both an image and text prompt.
        """
        if not self._processor:
            raise ValueError(
                f"Model '{self._model_name}' does not support images. Try a vision-capable Gemma variant."
            )

        # Load image
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)

        # Processor encodes both text + image
        inputs = self._processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        ).to(self._device)

        # Generate response
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode and clean
        completion = self._processor.decode(output.sequences[0], skip_special_tokens=True)
        answer_raw = completion
        answer = answer_raw

        return GemmaModelResponse(get_timestamp(), answer, answer_raw, output, self._tokenizer)

    def get_top_p_answer(
        self,
        distractor: Distractor | None,
        prompt_base: str,
        prompt_system: str,
        question_type: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GemmaModelResponse:
        """Top-p decoding (optional image)"""
        if distractor and distractor["modality"] == "img":
            print("Image")
            return self.generate_with_image(
                image_path=distractor["img_path"],
                system_prompt=prompt_system,
                user_prompt=prompt_base,
                max_new_tokens=max_tokens,
            )

        # Text-only fallback
        inputs = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).to(self._device)

        print("Text")
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        completion = self._tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        answer_raw = completion
        answer = completion[len(prompt_system) + len(prompt_base):].strip()

        return GemmaModelResponse(get_timestamp(), answer, answer_raw, output, self._tokenizer)