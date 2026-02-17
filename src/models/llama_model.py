import math
import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlamaTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput

from src.config import PATH_OFFLOAD, PATH_HF_CACHE
from src.models.model_utils import get_timestamp, get_api_key
from src.models.model import MODELS, LanguageModel, LanguageModelResponse

from src.prompters.prompt import Modality, Prompt


class LlamaModelResponse(LanguageModelResponse):
    _output: GenerateDecoderOnlyOutput
    _tokenizer: LlamaTokenizer

    def __init__(
        self,
        timestamp: str,
        answer: str,
        answer_raw: str,
        output: GenerateDecoderOnlyOutput,
        tokenizer: LlamaTokenizer
    ):
        super().__init__(
            timestamp=timestamp,
            answer_raw=answer_raw,
            answer=answer
        )
        self._output = output
        self._tokenizer = tokenizer

    def get_answer_prob(self, answer: str) -> float:
        token_ids = self._tokenizer(answer).input_ids
        if len(token_ids) - 1 > len(self._output.logits):
            return 0.0

        answer_log_prob = 0.0
        for i in range(len(token_ids) - 1):
            token_id = token_ids[i + 1]  # first token ID is BOS
            logits = self._output.logits[i]
            token_probs = torch.softmax(logits, dim=1).squeeze()
            token_prob = token_probs[token_id].item()
            if token_prob == 0.0:
                return 0.0
            answer_log_prob += math.log(token_prob)

        return math.exp(answer_log_prob)


class LlamaModel(LanguageModel):
    """Llama 3.2 Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "LlamaModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        # Setup access using HF login
        login(token=get_api_key("huggingface"))

        # Setup Device, Model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if MODELS[model_name]["8bit"]:
            raise ValueError(f"Unknown Model '{model_name}'")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
            )

        self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)

        # Setup Tokenizer
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
    ) -> LlamaModelResponse:
        """
        Query Llama model (with top-p decoding)

        :param prompt: the Prompt to query the model with
        :param max_tokens: the max output tokens
        :param temperature: the temperature to generate outputs with
        :param top_p: the probability to use for top_p decoding
        :return: a LlamaModelResponse with the model output
        """
        distractor = prompt["distractor"]
        if distractor is not None:
            if distractor["modality"] == Modality.IMAGE:
                raise ValueError("This model does not support image inputs!")

        if self._tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]}
            ]
            text_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Backup if no chat template exists
            text_prompt = prompt["system_prompt"] + " " + prompt["user_prompt"]

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
        answer = answer_raw[answer_raw.rfind("assistant") + len("assistant"):].strip()

        return LlamaModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=output,
            tokenizer=self._tokenizer
        )