from __future__ import annotations

import pandas as pd

from data.templates.question_templates import QUESTION_TEMPLATES
from data.templates.response_templates import A_TOKENS, B_TOKENS, YES_TOKENS, NO_TOKENS
from src.models.models import LanguageModelResponse
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor, Modality


class MoralChoiceScenario(Scenario):
    actions: list[str]


class MoralChoicePrompt(Prompt):
    question_format: str
    question_ordering: int


class MoralChoicePrompter(Prompter[MoralChoicePrompt]):
    def _generate_prompt(
        self,
        scenario: MoralChoiceScenario,
        question_format: str,
        question_ordering: int,
        distractor: Distractor | None = None
    ) -> MoralChoicePrompt:
        """
        Generate a prompt from the given information
        :param scenario: the scenario to prompt with
        :param question_format: the format of the question ("ab", "compare", "free")
        :param question_ordering: the ordering of the options (0, 1)
        :param distractor: the distractor to prompt with
        :return: a Prompt containing the given information
        """

        question_template = QUESTION_TEMPLATES[f"{question_format}_moralchoice"]
        prompt: MoralChoicePrompt = {
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": question_template["system"],
            "user_prompt": question_template["user"].format(
                scenario["context"],
                scenario["actions"][question_ordering],
                scenario["actions"][1 - question_ordering]
            ),
            "question_format": question_format,
            "question_ordering": question_ordering
        }
        return prompt

    def pre_process(
        self,
        scenario_series: pd.Series,
        question_format: str,
        distractor_series: pd.Series | None
    ) -> list[MoralChoicePrompt]:
        """
        Process scenario and distractor into prompts

        :param scenario_series: the pandas series with the scenario data
        :param question_format: the format of the question (ab, compare, reddit, free)
        :param distractor_series: the pandas series with the distractor data (optional)
        :return: a list of Prompts generated with the scenario and distractor
        """

        # Create Distractor
        distractor: Distractor | None = {
            "id": distractor_series["id"],
            "modality": distractor_series["modality"],
            "file_path": distractor_series["file_path"]
        } if distractor_series is not None else None

        # Create Scenario
        scenario: MoralChoiceScenario = {
            "id": scenario_series["id"],
            "context": scenario_series["context"],
            "actions": [scenario_series["action1"], scenario_series["action2"]]
        }

        # Generate prompts with all orderings
        prompts = []
        for question_ordering in [0, 1]:
            prompt = self._generate_prompt(
                scenario=scenario,
                distractor=distractor,
                question_format=question_format,
                question_ordering=question_ordering
            )
            prompts.append(prompt)
        return prompts

    def post_process(
        self,
        prompt: MoralChoicePrompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        """
        Process model response into result

        :param prompt: the Prompt used to query the model
        :param response: the model's response to the query
        :return: the results as a dict
        """

        if prompt["question_format"].startswith("ab"):
            # A/B question format
            action_tokens_dict = {
                "A": A_TOKENS,
                "B": B_TOKENS
            }
            action_mapping_dict = {
                "A": prompt["question_ordering"],
                "B": 1 - prompt["question_ordering"]
            }
        else:
            # YES/NO question format
            action_tokens_dict = {
                "YES": YES_TOKENS,
                "NO": NO_TOKENS
            }
            action_mapping_dict = {
                "YES": prompt["question_ordering"],
                "NO": 1 - prompt["question_ordering"]
            }
        # TODO: incorporate LLM-as-judge for open-ended generation

        result = {
            "model_id": self.model.get_model_id(),
            "question_format": prompt["question_format"],
            "question_ordering": prompt["question_ordering"],
            "system_prompt": prompt["system_prompt"],
            "user_prompt": prompt["user_prompt"],
            "eval_technique": "top_p_sampling",
            "eval_top_p": self.top_p,
            "eval_temperature": self.temperature,
            "scenario_id": prompt["scenario"]["id"],
            "distractor_id": prompt["distractor"]["id"] if prompt["distractor"] is not None else None,
            "answer_raw": response.answer_raw,
            "answer": response.answer,
            "action_0_prob": 0.0,
            "action_1_prob": 0.0
        }

        for action, tokens in action_tokens_dict.items():
            for token in tokens:
                result[f"action_{action_mapping_dict[action]}_prob"] += response.get_answer_prob(token)

        return result
