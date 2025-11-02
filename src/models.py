import os
import re
import sys
import time
import math
from abc import abstractmethod

import torch
import ai21
import cohere
import anthropic
import openai
import google.generativeai as palm
import Path

from google.api_core import retry
from typing import List, Dict
from datetime import datetime

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    AutoProcessor
)

from src.config import PATH_API_KEYS, PATH_HF_CACHE, PATH_OFFLOAD

from accelerate import Accelerator
from huggingface_hub import login

from PIL import Image

from src.prompters import Distractor

API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

####################################################################################
# MODELS DICT
####################################################################################
MODELS = dict(
    {
        "ai21/j2-grande-instruct": {
            "company": "AI21",
            "model_class": "AI21Model",
            "model_name": "j2-grande-instruct",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "ai21/j2-jumbo-instruct": {
            "company": "AI21",
            "model_class": "AI21Model",
            "model_name": "j2-jumbo-instruct",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "cohere/command-xlarge": {
            "company": "cohere",
            "model_class": "CohereModel",
            "model_name": "command-xlarge-nightly",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "cohere/command-medium": {
            "company": "cohere",
            "model_class": "CohereModel",
            "model_name": "command-medium-nightly",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": None,
        },
        "openai/text-ada-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-ada-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-babbage-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-babbage-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-curie-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-curie-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-davinci-001": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-davinci-001",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-davinci-002": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-davinci-002",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/text-davinci-003": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "text-davinci-003",
            "8bit": None,
            "likelihood_access": True,
            "endpoint": "Completion",
        },
        "openai/gpt-3.5-turbo": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-3.5-turbo",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": "ChatCompletion",
        },
        "openai/gpt-4": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-4",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": "ChatCompletion",
        },
        "openai/gpt-4.1-mini": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-4.1-mini",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": "ChatCompletion",
        },
        "anthropic/claude-v1.0": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-1.0",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-v1.2": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-1.2",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-v1.3": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-1.3",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-v2.0": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-2.0",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-instant-v1.0": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-instant-1.0",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "anthropic/claude-instant-v1.1": {
            "company": "anthropic",
            "model_class": "AnthropicModel",
            "model_name": "claude-instant-1.1",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": None,
        },
        "google/t5-small": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-small",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-base": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-base",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-large": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-large",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-xl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-xl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/t5-xxl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "t5-xxl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-small": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-small",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-base": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-base",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-large": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-large",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xxl": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xxl",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-small-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-small",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-base-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-base",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-large-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-large",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xl-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xl",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/flan-t5-xxl-8bit": {
            "company": "google",
            "model_class": "FlanT5Model",
            "model_name": "google/flan-t5-xxl",
            "8bit": True,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-regular-small": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-1.3b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-regular-large": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-30b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-max-small": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-max-1.3b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "meta/opt-iml-max-large": {
            "company": "meta",
            "model_class": "OptImlModel",
            "model_name": "facebook/opt-iml-max-30b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-560m": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-560m",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-1b1": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-1b1",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-1b7": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-1b7",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-3b": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-3b",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-7b1": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-7b1",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-7b1-mt": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-7b1-mt",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "bigscience/bloomz-7b1-p3": {
            "company": "bigscience",
            "model_class": "BloomZModel",
            "model_name": "bigscience/bloomz-7b1-p3",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/text-bison-001": {
            "company": "google",
            "model_class": "PalmModel",
            "model_name": "google/text-bison-001",
            "8bit": False,
            "likelihood_access": False,
            "endpoint": None,
        },
        "meta-llama/Llama-3.2-1B-Instruct": {
            "company": "meta",
            "model_class": "LlamaModel",
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-2-2b-it": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-2-2b-it",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
    }
)

####################################################################################
# HELPER FUNCTIONS
####################################################################################


def get_api_key(company_identifier: str) -> str:
    """
    Helper Function to retrieve API key from files
    """
    path_key = str(PATH_API_KEYS / f"{company_identifier}_key.txt")

    if os.path.exists(path_key):
        with open(path_key, encoding="utf-8") as f:
            key = f.read()
        return key

    raise ValueError(f"API KEY not available at: {path_key}")


def get_raw_likelihoods_from_answer(
    token_likelihoods: Dict[str, float], start: int = 0, end: int = None
) -> List[float]:
    """
    Helper Function to filter token_likelihood
    """

    if token_likelihoods[start][0] in [":", "\s", ""]:
        start += 1

    likelihoods = [
        likelihood
        for (token, likelihood) in token_likelihoods[start:end]
        if token not in ["<BOS_TOKEN>", "</s>"]
    ]

    return likelihoods


def get_timestamp():
    """
    Generate timestamp of format Y-M-D_H:M:S
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


####################################################################################
# MODEL WRAPPERS
####################################################################################


class LanguageModel:
    """ Generic LanguageModel Class"""
    
    def __init__(self, model_name):
        assert model_name in MODELS, f"Model {model_name} is not supported!"

        # Set some default model variables
        self._model_id = model_name
        self._model_name = MODELS[model_name]["model_name"]
        self._model_endpoint = MODELS[model_name]["endpoint"]
        self._company = MODELS[model_name]["company"]
        self._likelihood_access = MODELS[model_name]["likelihood_access"]

    def get_model_id(self):
        """Return model_id"""
        return self._model_id

    @abstractmethod
    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        """
        Gets greedy answer for prompt_base

        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :return:                answer string
        """
        pass

    @abstractmethod
    def get_top_p_answer(
        self,
        distractor: Distractor | None,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, any]:
        """
        Gets answer using sampling (based on top_p and temperature)

        :param distractor:      the distractor to inject
        :param prompt_base:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :param max_tokens       max tokens in answer
        :param temperature      temperature for top_p sampling
        :param top_p            top_p parameter
        :return:                answer string
        """
        pass


# ----------------------------------------------------------------------------------------------------------------------
# OPENAI MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------
class OpenAIModel(LanguageModel):
    """OpenAI API Wrapper"""
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OpenAIModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )

        api_key = get_api_key("openai")
        # openai.api_key = api_key
        self._client = openai.OpenAI(api_key=api_key)

    def _prompt_request(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logprobs: int = 1,
        stop: List = ["Human:", " AI:"],
        echo: bool = False,
        image_path: str = None,
    ):
        success = False
        t = 0

        while not success:
            try:
                # Dialogue Format
                user_content = [{"type": "text", "text": prompt_base}]
                if image_path:
                    # Append image to same user message
                    image_url = f"data:image/{Path(image_path).suffix[1:]};base64,{self._encode_image(image_path)}"
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})

                messages = [
                    {"role": "system", "content": f"{prompt_system[:-2]}"},
                    {"role": "user", "content": user_content},
                ]

                # Query ChatCompletion endpoint
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logprobs=True,
                    top_logprobs=20
                )

                # Set success flag
                success = True

            except Exception as e:
                print(e)
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS) - 1)

        return response

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int, image_path: str = None
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
            image_path=image_path
        )

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

        # (1) Top-P Sampling
        response = self._prompt_request(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=1,
            stop=["Human:", " AI:"],
            echo=False,
            image_path=image_path
        )

        completion = response.choices[0].message.content.strip()

        result["answer_raw"] = completion.strip()
        result["answer"] = completion.strip()

        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        token_probs = {
            "Yes": 0,
            "yes": 0,
            "No": 0,
            "no": 0,
            "A": 0,
            "a": 0,
            "B": 0,
            "b": 0,
            " Yes": 0,
            " No": 0,
            " yes": 0,
            " no": 0,
            " A": 0,
            " B": 0,
            " a": 0,
            " b": 0
        }
        for logprob in logprobs:
            if logprob.token in token_probs.keys():
                token_probs[logprob.token] = math.exp(logprob.logprob)

        result["token_prob_yes"] = token_probs["Yes"] + token_probs["yes"] + token_probs[" Yes"] + token_probs[" yes"]
        result["token_prob_no"] = token_probs["No"] + token_probs["no"] + token_probs[" No"] + token_probs[" no"]
        result["token_prob_a"] = token_probs["A"] + token_probs["a"] + token_probs[" A"] + token_probs[" a"]
        result["token_prob_b"] = token_probs["B"] + token_probs["b"] + token_probs[" B"] + token_probs[" b"]

        return result


# ----------------------------------------------------------------------------------------------------------------------
# ANTHROPIC MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------


class AnthropicModel(LanguageModel):
    """Anthropic API Wrapper"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "AnthropicModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("anthropic")
        self._anthropic_client = anthropic.Anthropic(api_key=api_key)

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Setup prompt according to Anthropic Format
        if self._model_name in ["claude-v2.0", "claude-instant-v1.1"]:
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt_system}{prompt_base}{anthropic.AI_PROMPT}"
        else:
            prompt = f"{anthropic.HUMAN_PROMPT} {prompt_system}{prompt_base}<result>YOUR CHOICE HERE</result>{anthropic.AI_PROMPT}"

        success = False
        t = 0

        while not success:
            try:
                # Prompt model for response
                response = self._anthropic_client.completions.create(
                    prompt=prompt,
                    model=self._model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens_to_sample=max_tokens,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                )
                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS))

        completion = response.completion.strip()
        result["answer_raw"] = completion

        # Anthropic Specific Post-Processing
        if "<result>" in response.completion:
            pattern = r"<result>([\s\S]*?)</result>"
            completion = re.findall(pattern, completion)

            if len(completion) == 1:
                completion = completion[0].strip()

        if "\n" in completion:
            completion = completion.split("\n")[0]

        if isinstance(completion, list):
            if len(completion) > 0:
                completion = completion[0]
            else:
                completion = ""

        result["answer"] = completion.strip()

        return result


# ----------------------------------------------------------------------------------------------------------------------
# AI21 MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------
class AI21Model(LanguageModel):
    """AI21 API Wrapper"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "AI21Model", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        api_key = get_api_key("ai21")
        ai21.api_key = api_key

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        return self.get_top_p_answer(
            prompt_base=prompt_base,
            prompt_system=prompt_system,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
        )

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        success = False
        t = 0

        while not success:
            try:
                response = ai21.Completion.execute(
                    model=self._model_name,
                    prompt=f"{prompt_system}{prompt_base}",
                    numResults=1,
                    maxTokens=max_tokens,
                    temperature=temperature,
                    topKReturn=0,
                    topP=top_p,
                    countPenalty={
                        "scale": 0,
                        "applyToNumbers": False,
                        "applyToPunctuations": False,
                        "applyToStopwords": False,
                        "applyToWhitespaces": False,
                        "applyToEmojis": False,
                    },
                    frequencyPenalty={
                        "scale": 0,
                        "applyToNumbers": False,
                        "applyToPunctuations": False,
                        "applyToStopwords": False,
                        "applyToWhitespaces": False,
                        "applyToEmojis": False,
                    },
                    presencePenalty={
                        "scale": 0,
                        "applyToNumbers": False,
                        "applyToPunctuations": False,
                        "applyToStopwords": False,
                        "applyToWhitespaces": False,
                        "applyToEmojis": False,
                    },
                    stopSequences=[],
                )
                success = True
            except:
                time.sleep(API_TIMEOUTS[t])
                t = min(t + 1, len(API_TIMEOUTS))

        completion = response.completions[0].data.text.strip()
        result["answer_raw"] = completion
        result["answer"] = completion

        return result


# ----------------------------------------------------------------------------------------------------------------------
# FLAN-T5 MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------

class FlanT5Model(LanguageModel):
    """Flan-T5 Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "FlanT5Model", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup Device, Model
        #self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if MODELS[model_name]["8bit"]:
            self._quantization_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True
            )

            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                quantization_config=self._quantization_config,
                load_in_8bit=MODELS[model_name]["8bit"],
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
            )
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path=self._model_name,
                cache_dir=PATH_HF_CACHE,
                device_map="auto",
                offload_folder=PATH_OFFLOAD,
            )#.to(self._device)

        self._device = next(self._model.parameters()).device

        # Setup Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

        self._token_ids = {
            "Yes": self._tokenizer("Yes").input_ids[0],
            "No": self._tokenizer("No").input_ids[0],
            "yes": self._tokenizer("yes").input_ids[0],
            "no": self._tokenizer("no").input_ids[0],
            "A": self._tokenizer("A").input_ids[0],
            "B": self._tokenizer("B").input_ids[0],
            "a": self._tokenizer("a").input_ids[0],
            "b": self._tokenizer("b").input_ids[0],
            " Yes": self._tokenizer(" Yes").input_ids[0],
            " No": self._tokenizer(" No").input_ids[0],
            " yes": self._tokenizer(" yes").input_ids[0],
            " no": self._tokenizer(" no").input_ids[0],
            " A": self._tokenizer(" A").input_ids[0],
            " B": self._tokenizer(" B").input_ids[0],
            " a": self._tokenizer(" a").input_ids[0],
            " b": self._tokenizer(" b").input_ids[0]
        }

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output
        completion = self._tokenizer.decode(
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
    ) -> any:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
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
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        ).strip()
        result["answer_raw"] = completion
        result["answer"] = completion

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


# ----------------------------------------------------------------------------------------------------------------------
# OPT-IML MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------

class OptImlModel(LanguageModel):
    """Meta OPT-IML Model Wrapper --> Access through HuggingFace Model Hub"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OptImlModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup Device, Model and Tokenizer
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE,
        ).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> OPT Repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt - 1 :].strip()
        result["answer"] = completion

        return result

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> OPT Repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt - 1 :].strip()
        result["answer"] = completion

        return result


# ----------------------------------------------------------------------------------------------------------------------
# BLOOMZ MODEL WRAPPER
# ----------------------------------------------------------------------------------------------------------------------


class BloomZModel(LanguageModel):
    """BigScience BloomZ Model Wrapper --> Access through HuggingFace Model Hub"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "BloomZModel", (
            f"Errorneous Model Instatiation for {model_name}"
        )

        # Setup Device, Model and Tokenizer
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self._model_name,
            cache_dir=PATH_HF_CACHE,
            torch_dtype="auto",
            device_map="auto",
            offload_folder=PATH_OFFLOAD,
        ).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
        )

    def get_greedy_answer(
        self, prompt_base: str, prompt_system: str, max_tokens: int
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> bloomz Repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt:].strip()
        result["answer"] = completion

        return result

    def get_top_p_answer(
        self,
        prompt_base: str,
        prompt_system: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        result = {
            "timestamp": get_timestamp(),
        }

        # Greedy Search
        input_ids = self._tokenizer(
            f"{prompt_system}{prompt_base}", return_tensors="pt"
        ).input_ids.to(self._device)
        response = self._model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            length_penalty=0,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Parse Output --> bloomz repeats prompt text before answer --> Cut it
        completion = self._tokenizer.decode(
            response.sequences[0], skip_special_tokens=True
        )
        result["answer_raw"] = completion
        len_prompt = len(f"{prompt_system}{prompt_base}")
        completion = completion[len_prompt:].strip()
        result["answer"] = completion

        return result


####################################################################################
# MODEL CREATOR
####################################################################################

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
    

# ----------------------------------------------------------------------------------------------------------------------
# Previous (text only Gemma model)
# class GemmaModel(LanguageModel):
#     """Gemma 2 Model Wrapper --> Access through HuggingFace Model Hub"""

#     def __init__(self, model_name: str):
#         super().__init__(model_name)
#         assert MODELS[model_name]["model_class"] == "GemmaModel", (
#             f"Errorneous Model Instatiation for {model_name}"
#         )

#         # Setup access using HF login
#         login(token=get_api_key("huggingface"))

#         # Setup Device, Model
#         #self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         if MODELS[model_name]["8bit"]:
#             raise ValueError(f"Unknown Model '{model_name}'")
#         else:
#             self._model = AutoModelForCausalLM.from_pretrained(
#                 pretrained_model_name_or_path=self._model_name,
#                 cache_dir=PATH_HF_CACHE,
#                 device_map="auto",
#                 offload_folder=PATH_OFFLOAD,
#             )#.to(self._device)

#         self._device = next(self._model.parameters()).device

#         # Setup Tokenizer
#         self._tokenizer = AutoTokenizer.from_pretrained(
#             pretrained_model_name_or_path=self._model_name, cache_dir=PATH_HF_CACHE
#         )

#         self._token_ids = {
#             "Yes": self._tokenizer("Yes").input_ids[1],
#             "No": self._tokenizer("No").input_ids[1],
#             "yes": self._tokenizer("yes").input_ids[1],
#             "no": self._tokenizer("no").input_ids[1],
#             "A": self._tokenizer("A").input_ids[1],
#             "B": self._tokenizer("B").input_ids[1],
#             "a": self._tokenizer("a").input_ids[1],
#             "b": self._tokenizer("b").input_ids[1],
#             " Yes": self._tokenizer(" Yes").input_ids[1],
#             " No": self._tokenizer(" No").input_ids[1],
#             " yes": self._tokenizer(" yes").input_ids[1],
#             " no": self._tokenizer(" no").input_ids[1],
#             " A": self._tokenizer(" A").input_ids[1],
#             " B": self._tokenizer(" B").input_ids[1],
#             " a": self._tokenizer(" a").input_ids[1],
#             " b": self._tokenizer(" b").input_ids[1]
#         }

#     def get_greedy_answer(
#         self, prompt_base: str, prompt_system: str, max_tokens: int, image_path: str = None
#     ) -> str:
#         result = {
#             "timestamp": get_timestamp(),
#         }

#         # Greedy Search
#         input_ids = self._tokenizer(
#             f"{prompt_system}{prompt_base}", return_tensors="pt"
#         ).input_ids.to(self._device)
#         response = self._model.generate(
#             input_ids,
#             max_new_tokens=max_tokens,
#             length_penalty=0,
#             output_scores=True,
#             return_dict_in_generate=True,
#         )

#         # Parse Output
#         completion = self._tokenizer.decode(
#             response.sequences[0], skip_special_tokens=True
#         ).strip()
#         result["answer_raw"] = completion
#         result["answer"] = completion

#         return result

#     def get_top_p_answer(
#         self,
#         prompt_base: str,
#         prompt_system: str,
#         max_tokens: int,
#         temperature: float,
#         top_p: float,
#     ) -> any:
#         result = {
#             "timestamp": get_timestamp(),
#         }

#         # Greedy Search
#         input_ids = self._tokenizer(
#             f"{prompt_system}{prompt_base}", return_tensors="pt"
#         ).input_ids.to(self._device)
#         response = self._model.generate(
#             input_ids,
#             max_new_tokens=1,
#             length_penalty=0,
#             do_sample=True,
#             top_p=top_p,
#             temperature=temperature,
#             output_scores=True,
#             output_logits=True,
#             return_dict_in_generate=True,
#         )

#         # Parse Output
#         completion = self._tokenizer.decode(
#             response.sequences[0], skip_special_tokens=True
#         ).strip()
#         result["answer_raw"] = completion
#         result["answer"] = completion[len(prompt_system) + len(prompt_base):]

#         probs = torch.softmax(response.logits[0], dim=1).squeeze()
        
#         result["token_prob_yes"] = probs[self._token_ids[" Yes"]].item() + probs[self._token_ids[" yes"]].item() 
#         + probs[self._token_ids["Yes"]].item() + probs[self._token_ids["yes"]].item()
#         result["token_prob_no"] = probs[self._token_ids[" No"]].item() + probs[self._token_ids[" no"]].item() 
#         + probs[self._token_ids["No"]].item() + probs[self._token_ids["no"]].item()
#         result["token_prob_a"] = probs[self._token_ids[" A"]].item() + probs[self._token_ids[" a"]].item() 
#         + probs[self._token_ids["A"]].item() + probs[self._token_ids["a"]].item()
#         result["token_prob_b"] = probs[self._token_ids[" B"]].item() + probs[self._token_ids[" b"]].item() 
#         + probs[self._token_ids["B"]].item() + probs[self._token_ids["b"]].item()

#         return result

def create_model(model_name):
    """Init Models from model_name only"""
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model_name)

    raise ValueError(f"Unknown Model '{model_name}'")
