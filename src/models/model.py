from __future__ import annotations

from abc import abstractmethod, ABC

import orjson
import pandas as pd

from src.config import PATH_DATA
from src.prompters.prompt import Prompt
from src.models.model_configs import MODELS


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
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> LanguageModelResponse:
        """
        Gets answer using sampling (based on top_p and temperature)

        :param distractor:      the distractor to inject
        :param user_prompt:     base prompt
        :param prompt_sytem:    system instruction for chat endpoint of OpenAI
        :param max_new_tokens       max tokens in answer
        :param temperature      temperature for top_p sampling
        :param top_p            top_p parameter
        :return:                answer string
        """
        pass


class BatchSubmitLanguageModel(LanguageModel, ABC):
    """Generic Batch Submit Language Model class"""
    _output_filename: str

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._output_filename = ""

    def set_filename(self, filename: str):
        self._output_filename = filename


class BatchRetrieveLanguageModel(LanguageModel, ABC):
    """Generic Batch Retrieve Language Model class"""
    _indices: dict[str, int]
    _lines: list[str]

    def __init__(self, model_name):
        super().__init__(model_name)
        self._indices = {}

    def load_data(self, response_filename: str):
        # Cache mapping of custom ID to line index
        with open(PATH_DATA / response_filename, 'r') as f:
            self._lines = f.readlines()
        for i, line in enumerate(self._lines):
            line_json = orjson.loads(line)
            self._indices[line_json["custom_id"]] = i
