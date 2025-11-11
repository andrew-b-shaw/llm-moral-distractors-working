import math

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlamaTokenizer

from src.config import PATH_OFFLOAD, PATH_HF_CACHE
from src.models.model_utils import get_timestamp, get_api_key
from src.models.models import MODELS, LanguageModel, LanguageModelResponse
from PIL import Image


class LlamaModelResponse(LanguageModelResponse):
    _output: any
    _tokenizer: LlamaTokenizer

    def __init__(
        self,
        timestamp: str,
        answer: str,
        answer_raw: str,
        output: any,
        tokenizer: LlamaTokenizer
    ):
        super().__init__(timestamp, answer, answer_raw)
        self._output = output
        print(type(output))
        self._tokenizer = tokenizer

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

    def query(
            self,
            user_prompt: str,
            system_prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            image_path: str = None,
    ) -> LlamaModelResponse:
        # Greedy Search
        text = f"{system_prompt}{user_prompt}"
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
        answer_raw = completion
        answer = completion[len(system_prompt) + len(user_prompt) - 1:]

        return LlamaModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=response,
            tokenizer=self._tokenizer
        )