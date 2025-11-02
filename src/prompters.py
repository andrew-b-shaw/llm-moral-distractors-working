from abc import abstractmethod
from typing import TypedDict, Callable, cast, TypeVar, Generic

import pandas as pd

from data.question_templates.question_templates import QuestionTemplate, QUESTION_TEMPLATES
from src.models import LanguageModel
from src.semantic_matching import token_to_action_matching


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
        pass

    @abstractmethod
    def post_process(
        self,
        prompt: AnyPrompt,
        response: dict[str, any]
    ) -> dict[str, any]:
        pass

    def prompt(
        self,
        scenario_series: pd.Series,
        question_type: str,
        distractor_series: pd.Series | None = None
    ):
        prompts = self.pre_process(
            scenario_series=scenario_series,
            distractor_series=distractor_series,
            distractor_modality=distractor_series["modality"] if distractor_series else None
        )

        responses = []
        for prompt in prompts:
            response = self.model.get_top_p_answer(
                distractor=prompt["distractor"],
                prompt_base=prompt["user_prompt"],
                prompt_system=prompt["system_prompt"],
                response_template=RESPONSE_TEMPLATES[question_type],
                max_tokens=self.max_tokens,
                temperature=self.eval_temp,
                top_p=self.eval_top_p
            )
            responses.append(response)

        results = []
        for prompt, response in zip(prompts, responses):
            result = self.post_process(prompt, response)
            results.append(result)
        return results


# ----------------------------------------------------------------------------------------------------------------------
# MORALCHOICE DATASET
# ----------------------------------------------------------------------------------------------------------------------
class MoralChoiceScenario(Scenario):
    actions: list[str]


class MoralChoicePrompt(Prompt):
    question_type: str
    question_ordering: int


class MoralChoicePrompter(Prompter[MoralChoicePrompt]):
    def generate_prompt(
        self,
        scenario: MoralChoiceScenario,
        question_type: str,
        question_ordering: int,
        distractor: Distractor | None = None
    ) -> MoralChoicePrompt:
        question_template = QUESTION_TEMPLATES[question_type]
        prompt: MoralChoicePrompt = {
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": question_template["system"],
            "user_prompt": question_template["user"].format(
                scenario["context"],
                scenario["actions"][question_ordering],
                scenario["actions"][1 - question_ordering]
            ),
            "question_type": question_type,
            "question_ordering": question_ordering
        }

        return prompt

    def pre_process(
        self,
        scenario_series: pd.Series,
        question_type: str,
        distractor_series: pd.Series | None,
        distractor_modality: str | None
    ) -> list[MoralChoicePrompt]:
        distractor: Distractor | None = {
            "modality": distractor_modality,
            "distractor_id": distractor_series["distractor_id"],
            "text": distractor_series["text"] if distractor_modality == "text" else None,
            "img": distractor_series["img_path"] if distractor_modality == "img" else None
        } if distractor_series else None

        scenario: MoralChoiceScenario = {
            "scenario_id": scenario_series["scenario_id"],
            "context": scenario_series["context"],
            "actions": [scenario_series["action1"], scenario_series["action2"]]
        }

        prompts = []
        for question_ordering in [0, 1]:
            prompt = self.generate_prompt(
                scenario=scenario,
                distractor=distractor,
                question_type=question_type,
                question_ordering=question_ordering
            )
            prompts.append(prompt)
        return prompts

    def post_process(
        self,
        prompt: MoralChoicePrompt,
        response: dict[str, any]
    ) -> dict[str, any]:
        if prompt["question_type"] == "ab":
            action_mapping_dict = {
                "A": prompt["question_ordering"],
                "B": 1 - prompt["question_ordering"]
            }
        else:
            action_mapping_dict = {
                "YES": prompt["question_ordering"],
                "NO": 1 - prompt["question_ordering"]
            }
        action_mapping = lambda x: action_mapping_dict[x]  # TODO: incorporate LLM-as-judge for open-ended generation

        # TODO: complete MoralChoice post-processing
        pass


# ----------------------------------------------------------------------------------------------------------------------
# REDDIT DATASET
# ----------------------------------------------------------------------------------------------------------------------
class RedditPrompter(Prompter[Prompt]):
    def generate_prompt(
        self,
        scenario: Scenario,
        question_type: str,
        distractor: Distractor | None,
    ) -> Prompt:
        question_template: QuestionTemplate = QUESTION_TEMPLATES[question_type]
        prompt: Prompt = {
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": question_template["system"],
            "user_prompt": question_template["user"].format(scenario["context"])
        }

        return prompt

    def pre_process(
        self,
        scenario_series: pd.Series,
        question_type: str,
        distractor_series: pd.Series | None,
        distractor_modality: str | None
    ) -> list[Prompt]:
        distractor: Distractor | None = {
            "modality": distractor_modality,
            "distractor_id": distractor_series["distractor_id"],
            "text": distractor_series["text"] if distractor_modality == "text" else None,
            "img": distractor_series["img_path"] if distractor_modality == "img" else None
        } if distractor_series else None

        scenario: Scenario = {
            "scenario_id": scenario_series["scenario_id"],
            "context": scenario_series["selftext"]
        }

        prompt = self.generate_prompt(
            scenario=scenario,
            question_type=question_type,
            distractor=distractor
        )
        return [prompt]

    def post_process(
            self,
            prompt: Prompt,
            response: dict[str, any]
    ) -> dict[str, any]:
        # TODO: complete reddit post-processing
        pass