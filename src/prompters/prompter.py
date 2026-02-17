from __future__ import annotations

import pandas as pd

from abc import abstractmethod, ABC
from typing import TypeVar, Generic

from src.models.model import LanguageModel, LanguageModelResponse
from src.prompters.prompt import Prompt

AnyPrompt = TypeVar("AnyPrompt", bound=Prompt)


class Prompter(Generic[AnyPrompt]):
    model: LanguageModel
    max_tokens: int
    temperature: float
    top_p: float

    def __init__(
        self,
        model: LanguageModel,
        max_tokens: int,
        temperature: float,
        top_p: float
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def pre_process(
        self,
        scenario_series: pd.Series,
        question_format: str,
        distractor_series: pd.Series | None
    ) -> list[AnyPrompt]:
        """
        Process scenario and distractor into prompts

        :param scenario_series: the pandas series with the scenario data
        :param question_format: the question format (ab, compare, reddit, free)
        :param distractor_series: the pandas series with the distractor data (optional)
        :return: a list of Prompts generated with the scenario and distractor
        """
        pass

    @abstractmethod
    def post_process(
        self,
        prompt: AnyPrompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        """
        Process model response into result

        :param prompt: the Prompt used to query the model
        :param response: the model's response to the query
        :return: the results as a dict
        """
        pass

    def prompt(
        self,
        question_format: str,
        scenario_series: pd.Series,
        distractor_series: pd.Series | None = None
    ) -> list[dict[str, any]]:
        """
        Prompts the model with the given scenario and distractor data

        :param scenario_series: the pandas series with the scenario data
        :param question_format: the format of the question (ab, compare, reddit, free)
        :param distractor_series: the pandas series with the distractor data
        :return: the results from the generated prompts
        """

        # Pre-process prompts
        prompts = self.pre_process(
            question_format=question_format,
            scenario_series=scenario_series,
            distractor_series=distractor_series,
        )

        # Query model with prompts
        responses = []
        for i, prompt in enumerate(prompts):
            response = self.model.query(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            responses.append(response)

        # Post-process responses
        results = []
        for prompt, response in zip(prompts, responses):
            result = self.post_process(prompt, response)
            results.append(result)
        return results


class BatchPrompter(Prompter[AnyPrompt], ABC):
    """Generic Batch Prompter class"""

    def __init__(
        self,
        model: LanguageModel,
        max_tokens: int,
        temperature: float,
        top_p: float
    ):
        super().__init__(model, max_tokens, temperature, top_p)
        self._submit_filename = None
        self._retrieve_filename = None

    def set_submit_filename(self, filename: str):
        self._submit_filename = filename

    def set_retrieve_filename(self, filename: str):
        self._retrieve_filename = filename

