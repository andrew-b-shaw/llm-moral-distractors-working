from abc import abstractmethod
from typing import TypedDict, TypeVar, Generic

import pandas as pd

from data.templates.question_templates import QuestionTemplate, QUESTION_TEMPLATES
from src.models.models import LanguageModel


class Distractor(TypedDict):
    modality: str
    distractor_id: str
    text: str | None
    img: str | None


class Scenario(TypedDict):
    scenario_id: str
    context: str


class Prompt(TypedDict):
    scenario: Scenario
    distractor: Distractor | None
    system_prompt: str
    user_prompt: str


AnyPrompt = TypeVar("AnyPrompt", bound=Prompt)


class Prompter(Generic[AnyPrompt]):
    model: LanguageModel
    max_tokens: int
    eval_temp: float
    eval_top_p: float

    def __init__(
        self,
        model: LanguageModel,
        max_tokens: int,
        eval_temp: float,
        eval_top_p: float
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.eval_temp = eval_temp
        self.eval_top_p = eval_top_p

    @abstractmethod
    def pre_process(
        self,
        scenario_series: pd.Series,
        question_type: str,
        distractor_series: pd.Series | None,
        distractor_modality: str | None
    ) -> list[AnyPrompt]:
        """
        Process scenario and distractor into prompts

        :param scenario_series: the pandas series with the scenario data
        :param question_type: the question type (ab, compare, reddit, free)
        :param distractor_series: the pandas series with the distractor data
        :param distractor_modality: the modality of the distractor
        :return: a list of Prompts generated with the scenario and distractor
        """
        pass

    @abstractmethod
    def post_process(
        self,
        prompt: AnyPrompt,
        response: dict[str, any]
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
        scenario_series: pd.Series,
        question_type: str,
        distractor_series: pd.Series | None = None
    ):
        """
        Prompts the model with the given scenario and distractor data

        :param scenario_series: the pandas series with the scenario data
        :param question_type: the question type (ab, compare, reddit, free)
        :param distractor_series: the pandas series with the distractor data
        :return: the results from the generated prompts
        """

        # Pre-process prompts
        prompts = self.pre_process(
            scenario_series=scenario_series,
            distractor_series=distractor_series,
            distractor_modality=distractor_series["modality"] if distractor_series else None
        )

        # TODO: join distractor with scenario

        # Query model with prompts
        responses = []
        for prompt in prompts:
            response = self.model.get_top_p_answer(
                distractor=prompt["distractor"],
                prompt_base=prompt["user_prompt"],
                prompt_system=prompt["system_prompt"],
                question_type=question_type,
                max_tokens=self.max_tokens,
                temperature=self.eval_temp,
                top_p=self.eval_top_p
            )
            responses.append(response)

        # Post-process responses
        results = []
        for prompt, response in zip(prompts, responses):
            result = self.post_process(prompt, response)
            results.append(result)
        return results
