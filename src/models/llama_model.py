import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from src.config import PATH_OFFLOAD, PATH_HF_CACHE
from src.models.model_utils import get_timestamp, get_api_key
from src.models.models import MODELS, LanguageModel
from PIL import Image


class LlamaModel(LanguageModel):
    """Llama 3.2 Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "LlamaModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup access using HF login
        login(token=get_api_key("huggingface"))

        # Setup Device, Model
        #self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if MODELS[model_name]["8bit"]:
            raise ValueError(f"Unknown Model '{model_name}'")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
            )#.to(self._device)

        self._device = next(self._model.parameters()).device

        self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)

        # Setup Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

        self._token_ids = {
            "Yes": self._tokenizer("Yes").input_ids[1],
            "No": self._tokenizer("No").input_ids[1],
            "yes": self._tokenizer("yes").input_ids[1],
            "no": self._tokenizer("no").input_ids[1],
            "A": self._tokenizer("A").input_ids[1],
            "B": self._tokenizer("B").input_ids[1],
            "a": self._tokenizer("a").input_ids[1],
            "b": self._tokenizer("b").input_ids[1],
            " Yes": self._tokenizer(" Yes").input_ids[1],
            " No": self._tokenizer(" No").input_ids[1],
            " yes": self._tokenizer(" yes").input_ids[1],
            " no": self._tokenizer(" no").input_ids[1],
            " A": self._tokenizer(" A").input_ids[1],
            " B": self._tokenizer(" B").input_ids[1],
            " a": self._tokenizer(" a").input_ids[1],
            " b": self._tokenizer(" b").input_ids[1]
        }

    def get_greedy_answer(
            self, prompt_base: str, prompt_system: str, max_tokens: int, image_path: str = None
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        text = f"{prompt_system}{prompt_base}"
        if image_path:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self._device)
        else:
            inputs = self._processor(
                text=text,
                return_tensors="pt"
            ).to(self._device)

        # Greedy Search
        response = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output
        completion = self._processor.decode(
            response.sequences[0], skip_special_tokens=True
        ).strip()
        result["answer_raw"] = completion
        result["answer"] = completion

        return result

    def get_top_p_answer(
            self,
            prompt_base: str,
            prompt_system: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            image_path: str = None,
    ) -> any:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        text = f"{prompt_system}{prompt_base}"
        if image_path:
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(self._device)
        else:
            inputs = self._processor(
                text=text,
                return_tensors="pt"
            ).to(self._device)

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

        # Parse Output
        completion = self._processor.decode(
            response.sequences[0], skip_special_tokens=True
        ).strip()
        result["answer_raw"] = completion
        result["answer"] = completion[len(prompt_system) + len(prompt_base) - 1:]

        probs = torch.softmax(response.logits[0], dim=1).squeeze()

        result["token_prob_yes"] = probs[self._token_ids[" Yes"]].item() + probs[self._token_ids[" yes"]].item()
        + probs[self._token_ids["Yes"]].item() + probs[self._token_ids["yes"]].item()
        result["token_prob_no"] = probs[self._token_ids[" No"]].item() + probs[self._token_ids[" no"]].item()
        + probs[self._token_ids["No"]].item() + probs[self._token_ids["no"]].item()
        result["token_prob_a"] = probs[self._token_ids[" A"]].item() + probs[self._token_ids[" a"]].item()
        + probs[self._token_ids["A"]].item() + probs[self._token_ids["a"]].item()
        result["token_prob_b"] = probs[self._token_ids[" B"]].item() + probs[self._token_ids[" b"]].item()
        + probs[self._token_ids["B"]].item() + probs[self._token_ids["b"]].item()

        return result