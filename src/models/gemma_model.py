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

class GemmaModel(LanguageModel):
    """Gemma 3 Model Wrapper --> Supports text + image prompts"""

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
            self._processor = AutoProcessor.from_pretrained(self._model_name, cache_dir=PATH_HF_CACHE)
        except Exception as e:
            self._processor = None
            print(e)

        self._device = next(self._model.parameters()).device

    def _generate_with_image(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
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

        # pipe = pipeline(
        #     "image-text-to-text",
        #     model=self._model.get_model_id(),
        #     device=self._device,
        #     torch_dtype=torch.bfloat16,
        # )
        # output = pipe(
        #     image_path,
        #     text=f"{system_prompt} <image_soft_token> {user_prompt}"
        # )

        # Processor encodes both text + image
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        # prompt = self._processor.apply_chat_template(messages)
        prompt = f"{system_prompt} <start_of_image> {user_prompt}"
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
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self._tokenizer.eos_token_id,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True
            )

        # Decode and clean
        answer_raw = self._processor.decode(output.sequences[0], skip_special_tokens=True)
        answer = answer_raw[len(prompt):].strip()

        return GemmaModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=output,
            tokenizer=self._tokenizer
        )

    def query(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        distractor: Distractor | None = None
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
        if distractor is not None:
            if distractor["modality"] == Modality.IMAGE:
                image_path = f"{os.path.abspath(os.getcwd())}/data/{distractor["file_path"]}"
                return self._generate_with_image(
                    image_path=image_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                text_path = f"{os.path.abspath(os.getcwd())}/data/{distractor["file_path"]}"
                with open(text_path, 'r') as f:
                    distractor_text = f.read()
                    user_prompt = f"{distractor_text} {user_prompt}"

        # Text-only fallback
        prompt = f"{system_prompt} {user_prompt}"
        inputs = self._tokenizer(
            prompt,
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
        answer = answer_raw[len(prompt):].strip()

        return GemmaModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer,
            output=output,
            tokenizer=self._tokenizer
        )