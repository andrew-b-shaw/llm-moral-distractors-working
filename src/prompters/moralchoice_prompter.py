"""Prompter for the MoralChoice benchmark.

Constructs A/B forced-choice prompts with both answer orderings per scenario,
injects textual or visual distractors with temporal distance, and computes
per-action probabilities by marginalizing over token variants (e.g. 'A', ' a', '[A]')."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from data.templates.question_templates import QUESTION_TEMPLATES
from data.templates.response_templates import A_TOKENS, B_TOKENS, YES_TOKENS, NO_TOKENS
from src.config import PATH_DISTRACTORS
from src.models.model import LanguageModelResponse
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor, Modality, ImagePosition


class MoralChoiceScenario(Scenario):
    actions: list[str]


class MoralChoicePrompt(Prompt):
    question_format: str
    question_ordering: int


class MoralChoicePrompter(Prompter[MoralChoicePrompt]):
    def _generate_prompt(
        self,
        question_format: str,
        question_ordering: int,
        scenario: MoralChoiceScenario,
        distractor: Optional[Distractor] = None
    ) -> MoralChoicePrompt:
        """
        Generate a prompt from the given information
        :param question_format: the MoralChoice question format ("ab" or "compare")
        :param question_ordering: the ordering of the options (0, 1)
        :param scenario: the Scenario to prompt with
        :param distractor: the Distractor to prompt with
        :return: a Prompt containing the given information
        """

        question_template = QUESTION_TEMPLATES[f"{question_format}_moralchoice"]
        context = scenario["context"]
        # Inject distractor into the scenario context with temporal distance
        # (e.g. "{distractor text}. Later, {scenario}") to add another layer
        # of moral irrelevance between the distractor and the dilemma.
        if distractor:
            if distractor["modality"] == Modality.TEXT:
                file_path = PATH_DISTRACTORS / distractor["file_path"]
                with open(file_path, "r", encoding="utf-8") as f:
                    distractor_text = f.read().strip()
                if distractor_text:
                    context = f"{distractor_text} Later, {context[0].lower() + context[1:]}"
            else:
                context = f"You see the scene in the image. {context}"

        prompt: MoralChoicePrompt = {
            "id": f"{scenario["id"]}" +
                  f"_{distractor["id"] if distractor else "none"}" +
                  f"_{question_ordering}" +
                  f"_{question_format}",
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": question_template["system"],
            "user_prompt": question_template["user"].format(
                context,
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
        distractor_series: Optional[pd.Series]
    ) -> list[MoralChoicePrompt]:
        """
        Process scenario and distractor into prompts

        :param scenario_series: the pandas series with the scenario data
        :param question_format: the MoralChoice question format ("ab" or "compare")
        :param distractor_series: the pandas series with the distractor data (optional)
        :return: a list of Prompts generated with the scenario and distractor
        """

        # Create Distractor
        distractor: Optional[Distractor] = {
            "id": distractor_series["id"],
            "modality": distractor_series["modality"],
            "file_path": distractor_series["file_path"],
            "position": ImagePosition.BEFORE_USER
        } if distractor_series is not None else None

        # Create Scenario
        scenario: MoralChoiceScenario = {
            "id": scenario_series["id"],
            "context": scenario_series["context"],
            "actions": [scenario_series["action1"], scenario_series["action2"]]
        }

        # Generate prompts with both answer orderings (action1 as A then as B)
        # to control for position bias; results are averaged across orderings.
        prompts = []
        for question_ordering in [0, 1]:
            prompt = self._generate_prompt(
                question_format=question_format,
                question_ordering=question_ordering,
                scenario=scenario,
                distractor=distractor
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

        # Map model output tokens back to canonical action indices (a1, a2).
        # question_ordering tracks which action was presented as option A:
        #   ordering=0 -> A=action1, B=action2
        #   ordering=1 -> A=action2, B=action1
        if prompt["question_format"].startswith("ab"):
            action_tokens_dict = {
                "A": A_TOKENS,
                "B": B_TOKENS
            }
            action_mapping_dict = {
                "A": prompt["question_ordering"],
                "B": 1 - prompt["question_ordering"]
            }
        else:
            action_tokens_dict = {
                "YES": YES_TOKENS,
                "NO": NO_TOKENS
            }
            action_mapping_dict = {
                "YES": prompt["question_ordering"],
                "NO": 1 - prompt["question_ordering"]
            }

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
            "a1_prob": 0.0,
            "a2_prob": 0.0
        }

        for action, tokens in action_tokens_dict.items():
            for token in tokens:
                result[f"a{action_mapping_dict[action] + 1}_prob"] += response.get_answer_prob(token)

        return result


class MoralChoiceBatchSubmitPrompter(MoralChoicePrompter):
    def post_process(
            self,
            prompt: Prompt,
            response: LanguageModelResponse
    ) -> dict[str, any]:
        return {}
