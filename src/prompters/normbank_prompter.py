"""Prompter for the Norm Bank benchmark.

Evaluates everyday situations as good, acceptable, or wrong. Distractors are
prepended to the system prompt (not the scenario) since the model is judging
others' actions rather than choosing its own. Marginal answer probabilities are
computed over token variants of each judgement category.

Verdict probability scoring uses a per-scenario RANDOMIZED LETTER MAPPING (good/
ok/bad -> A/B/C) instead of scoring "It's good"/"It's acceptable"/"It's wrong"
directly. This matters because those phrasings only diverge at their FIRST content
token ("good"/"acceptable"/"wrong") when they share the same "It's"/"It is" style,
but the model only ever generates ONE completion. get_answer_prob() re-uses that one
sampled generation's logits for every candidate string, so once a candidate's tokens
diverge from what was actually sampled, later logits were computed conditioned on
the ACTUAL (different) prefix -- not the candidate's -- silently biasing that
candidate's probability toward 0 regardless of the model's true confidence.
Single-letter answers side-step this: any two letters diverge only at their one
shared/final token, so the logits used are always conditioned on a prefix that was,
in fact, actually generated.

The mapping is seeded from the scenario id (not resampled per query) so that the
baseline and every distractor condition for the same scenario share the same
mapping -- otherwise the randomization itself would leak noise into the
baseline-vs-distractor comparison this whole codebase exists to measure."""

from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from data.templates.question_templates import QUESTION_TEMPLATES
from data.templates.response_templates import LETTER_TOKENS
from src.config import PATH_DISTRACTORS
from src.models.model import LanguageModelResponse
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor, Modality, ImagePosition


class NormBankPrompt(Prompt):
    letter_mapping: dict[str, str]  # judgement ("good"/"ok"/"bad") -> assigned letter


class NormBankPrompter(Prompter[NormBankPrompt]):
    # Canonical order: fixed regardless of the randomized letter assignment, so
    # result columns (good_prob, ok_prob, bad_prob) stay stable across scenarios.
    _JUDGEMENT_ORDER = ("good", "ok", "bad")
    _JUDGEMENT_DESCRIPTIONS = {
        "good": "It's good.",
        "ok": "It's acceptable.",
        "bad": "It's wrong.",
    }
    _LETTERS = ("A", "B", "C")

    def _build_letter_mapping(self, scenario_id: str) -> dict[str, str]:
        """
        Randomly (but deterministically, given scenario_id) assign a letter to each
        judgement. Seeded on scenario_id so the mapping is stable across the
        baseline and every distractor condition run for that scenario.
        """
        rng = random.Random(f"normbank:{scenario_id}")
        shuffled_letters = list(self._LETTERS)
        rng.shuffle(shuffled_letters)
        return dict(zip(self._JUDGEMENT_ORDER, shuffled_letters))

    def _build_choices_block(self, letter_mapping: dict[str, str]) -> str:
        # Present choices in letter order (A, B, C) for readability, even though
        # which judgement maps to which letter is randomized.
        letter_to_judgement = {letter: judgement for judgement, letter in letter_mapping.items()}
        lines = [
            f"{letter}) {self._JUDGEMENT_DESCRIPTIONS[letter_to_judgement[letter]]}"
            for letter in self._LETTERS
        ]
        return "\n".join(lines)

    def _generate_prompt(
        self,
        scenario: Scenario,
        distractor: Optional[Distractor] = None
    ) -> NormBankPrompt:
        """
        Generate a prompt from the given information
        :param scenario: the Scenario to prompt with
        :param distractor: the Distractor to prompt with
        :return: a Prompt containing the given information
        """

        question_template = QUESTION_TEMPLATES["normbank"]
        context = scenario["context"]
        letter_mapping = self._build_letter_mapping(scenario["id"])
        system = question_template["system"].format(choices=self._build_choices_block(letter_mapping))
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

        prompt: NormBankPrompt = {
            "id": f"{scenario["id"]}_{distractor["id"] if distractor else "none"}",
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": system,
            "user_prompt": question_template["user"].format(context),
            "letter_mapping": letter_mapping,
        }
        return prompt

    def pre_process(
        self,
        scenario_series: pd.Series,
        question_format: str,
        distractor_series: Optional[pd.Series]
    ) -> list[NormBankPrompt]:
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
        prompt: NormBankPrompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        """
        Process model response into result

        :param prompt: the Prompt used to query the model
        :param response: the model's response to the query
        :return: the results as a dict
        """

        # Score each judgement's marginal probability via the letter this scenario
        # assigned it (not the "It's good"/"It's wrong"/"It's acceptable" phrase --
        # see module docstring).
        action_tokens_dict = {
            judgement: LETTER_TOKENS[self._LETTERS.index(letter)]
            for judgement, letter in prompt["letter_mapping"].items()
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
