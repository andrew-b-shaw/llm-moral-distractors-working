import torch
from PIL import Image
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from src.config import PATH_HF_CACHE, PATH_OFFLOAD
from src.models.model_utils import get_timestamp, get_api_key
from src.models.models import LanguageModel, MODELS


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
            torch_dtype=torch.float16,
        )

        # Tokenizer + (optional) Processor
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)

        try:
            # Vision-capable Gemma models have a processor
            self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        except Exception:
            self._processor = None

        self._device = next(self._model.parameters()).device

        # Token mappings
        self._token_ids = {
            k: self._tokenizer(k).input_ids[1]
            for k in [
                "Yes","No","yes","no","A","B","a","b",
                " Yes"," No"," yes"," no"," A"," B"," a"," b"
            ]
        }

    def generate_with_image_and_text(
            self,
            image_path: str,
            text_prompt: str,
            max_new_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.9,
    ) -> str:
        """
        Generate a response from Gemma with both an image and text prompt.
        """
        if not self._processor:
            raise ValueError(
                f"Model '{self._model_name}' does not support images. "
                "Try a vision-capable Gemma variant."
            )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Processor encodes both text + image
        inputs = self._processor(
            text=text_prompt,
            images=image,
            return_tensors="pt"
        ).to(self._device)

        # Generate response
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode and clean
        completion = self._processor.decode(outputs[0], skip_special_tokens=True)
        return completion.strip()

    def get_greedy_answer(
            self, prompt_base: str, prompt_system: str, max_tokens: int, image_path: str = None
    ) -> str:
        """Greedy decoding (optional image)"""
        if image_path:
            # Use multimodal method
            return self.generate_with_image_and_text(
                image_path=image_path,
                text_prompt=f"{prompt_system}{prompt_base}",
                max_new_tokens=max_tokens,
                temperature=0,
                top_p=1.0,
            )

        # Text-only fallback
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)

        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        completion = self._tokenizer.decode(response[0], skip_special_tokens=True).strip()
        return completion

    def get_top_p_answer(
            self,
            prompt_base: str,
            prompt_system: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            image_path: str = None,
    ) -> any:
        """Top-p decoding (optional image)"""
        result = {"timestamp": get_timestamp()}

        if image_path:
            result["answer"] = self.generate_with_image_and_text(
                image_path=image_path,
                text_prompt=f"{prompt_system}{prompt_base}",
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            result["answer_raw"] = result["answer"]
            return result

        # Text-only fallback
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)

        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self._tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        completion = self._tokenizer.decode(response.sequences[0], skip_special_tokens=True)
        result["answer_raw"] = completion
        result["answer"] = completion[len(prompt_system) + len(prompt_base):].strip()
        return result