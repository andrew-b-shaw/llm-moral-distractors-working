"""Prompter for the Norm Bank benchmark.

Evaluates everyday situations as good, acceptable, or wrong. Distractors are
prepended to the system prompt (not the scenario) since the model is judging
others' actions rather than choosing its own. Marginal answer probabilities are
computed over token variants of each judgement category."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from data.templates.question_templates import QUESTION_TEMPLATES
from data.templates.response_templates import GOOD_TOKENS, OK_TOKENS, BAD_TOKENS
from src.config import PATH_DISTRACTORS
from src.models.model import LanguageModelResponse
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor, Modality, ImagePosition


class NormBankPrompter(Prompter[Prompt]):
    def _generate_prompt(
        self,
        scenario: Scenario,
        distractor: Optional[Distractor] = None
    ) -> Prompt:
        """
        Generate a prompt from the given information
        :param scenario: the Scenario to prompt with
        :param distractor: the Distractor to prompt with
        :return: a Prompt containing the given information
        """

        question_template = QUESTION_TEMPLATES["normbank"]
        context = scenario["context"]
        system = question_template["system"]
        # For Norm Bank (and r/AITA), distractors are prepended to the system prompt
        # rather than the scenario, since the model is judging others' actions.
        if distractor:
            if distractor["modality"] == Modality.TEXT:
                file_path = PATH_DISTRACTORS / distractor["file_path"]
                with open(file_path, "r", encoding="utf-8") as f:
                    distractor_text = f.read().strip()
                if distractor_text:
                    system = f"{distractor_text} Later, {system[0].lower() + system[1:]}"
            else:
                system = f"You see the scene in the image. {system}"

        prompt: Prompt = {
            "id": f"{scenario["id"]}_{distractor["id"] if distractor else "none"}",
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": system,
            "user_prompt": question_template["user"].format(context)
        }
        return prompt

    def pre_process(
        self,
        scenario_series: pd.Series,
        question_format: str,
        distractor_series: Optional[pd.Series]
    ) -> list[Prompt]:
        """
        Process scenario and distractor into prompts

        :param scenario_series: the pandas series with the scenario data
        :param question_format: the Norm Bank question format ("normbank")
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
        scenario: Scenario = {
            "id": scenario_series["id"],
            "context": scenario_series["context"]
        }

        return [self._generate_prompt(scenario, distractor)]

    def post_process(
        self,
        prompt: Prompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        """
        Process model response into result

        :param prompt: the Prompt used to query the model
        :param response: the model's response to the query
        :return: the results as a dict
        """

        action_tokens_dict = {
            "good": GOOD_TOKENS,
            "ok": OK_TOKENS,
            "bad": BAD_TOKENS
        }

        result = {
            "model_id": self.model.get_model_id(),
            "system_prompt": prompt["system_prompt"],
            "user_prompt": prompt["user_prompt"],
            "eval_technique": "top_p_sampling",
            "eval_top_p": self.top_p,
            "eval_temperature": self.temperature,
            "scenario_id": prompt["scenario"]["id"],
            "distractor_id": prompt["distractor"]["id"] if prompt["distractor"] is not None else None,
            "answer_raw": response.answer_raw,
            "answer": response.answer,
            "good_prob": 0.0,
            "ok_prob": 0.0,
            "bad_prob": 0.0
        }

        for action, tokens in action_tokens_dict.items():
            for token in tokens:
                result[f"{action}_prob"] += response.get_answer_prob(token)

        return result


class NormBankBatchSubmitPrompter(NormBankPrompter):
    def post_process(
            self,
            prompt: Prompt,
            response: LanguageModelResponse
    ) -> dict[str, any]:
        return {}
