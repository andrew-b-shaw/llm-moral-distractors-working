from __future__ import annotations

import os
import re
import sys
import time
import math
from abc import abstractmethod

from src.prompters.prompt import Distractor

API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

####################################################################################
# MODELS DICT
####################################################################################
MODELS = dict(
    {
        "openai/gpt-5": {
            "company": "openai",
            "model_class": "OpenAIModel",
            "model_name": "gpt-5",
            "8bit": None,
            "likelihood_access": False,
            "endpoint": "ChatCompletion",
        },
        "Qwen/Qwen2-VL-2B-Instruct": {
            "company": "AliBaba",
            "model_class": "QwenVLModel",
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None
        },
        "Qwen/Qwen3-1.7B": {
            "company": "AliBaba",
            "model_class": "QwenModel",
            "model_name": "Qwen/Qwen3-1.7B",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None
        },
        "Qwen/Qwen3-1.7B-Base": {
            "company": "AliBaba",
            "model_class": "QwenModel",
            "model_name": "Qwen/Qwen3-1.7B-Base",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None
        },
        "meta-llama/Llama-3.2-1B-Instruct": {
            "company": "meta",
            "model_class": "LlamaModel",
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-1b-pt": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-1b-pt",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-1b-it": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-1b-it",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-4b-pt": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-4b-pt",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-4b-it": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-4b-it",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-12b-pt": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-12b-pt",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-12b-it": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-12b-it",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-27b-pt": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-27b-pt",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "google/gemma-3-27b-it": {
            "company": "google",
            "model_class": "GemmaModel",
            "model_name": "google/gemma-3-27b-it",
            "8bit": False,
            "likelihood_access": True,
            "endpoint": None,
        },
        "ollama/gemma3-4b": {
            "company": "ollama",
            "model_class": "OllamaModel",
            "model_name": "ollama/gemma3-4b",
            "ollama_model": "gemma3:4b",
            "8bit": False,
            "likelihood_access": False,
            "endpoint": "ollama",
        },
    }
)

####################################################################################
# MODEL WRAPPERS
####################################################################################
class LanguageModelResponse:
    """Generic LanguageModelResponse Class"""

    timestamp: str
    answer: str
    answer_raw: str

    def __init__(self, timestamp: str, answer_raw: str, answer: str):
        self.timestamp = timestamp
        self.answer_raw = answer_raw
        self.answer = answer

    @abstractmethod
    def get_answer_prob(self, answer: str) -> float:
        """
        Returns probability that the output **starts** with given string

        :param answer: the string to calculate the probability of
        :return: the probability that the output **starts** with the given string
        """
        pass


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
    def query(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        distractor: Distractor | None = None
    ) -> LanguageModelResponse:
        """
        Gets answer using sampling (based on top_p and temperature)

        :param distractor:      the distractor to inject
        :param user_prompt:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :param max_tokens       max tokens in answer
        :param temperature      temperature for top_p sampling
        :param top_p            top_p parameter
        :return:                answer string
        """
        pass

