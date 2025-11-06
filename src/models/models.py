import os
import re
import sys
import time
import math
from abc import abstractmethod
from src.models.openai_model import OpenAIModel
from src.models.gemma_model import GemmaModel
from src.models.llama_model import LlamaModel

from src.prompters.prompter import Distractor

API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

####################################################################################
# MODELS DICT
####################################################################################
MODELS = dict(
    {
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
        question_type: str,
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


####################################################################################
# MODEL CREATOR
####################################################################################
def create_model(model_name):
    """Init Models from model_name only"""
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model_name)

    raise ValueError(f"Unknown Model '{model_name}'")
