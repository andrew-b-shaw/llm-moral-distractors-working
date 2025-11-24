from __future__ import annotations

import re

import pandas as pd

from data.templates.question_templates import QuestionTemplate, QUESTION_TEMPLATES
from src.classifier import ME2BERTScorer
from src.config import PATH_DISTRACTORS
from src.models.models import LanguageModelResponse, LanguageModel
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor, Modality, Position


class RedditPrompter(Prompter[Prompt]):
    _QUESTION_TYPE = "reddit"
    _VALID_VERDICTS = ("YTA", "NTA", "ESH", "NAH", "INFO")
    _VERDICT_REGEX = re.compile(r"(?i)\bVerdict\b\s*[:\-]\s*(YTA|NTA|ESH|NAH|INFO)\b")
    _REASONING_REGEX = re.compile(r"(?is)\b(?:Reasoning|Reason|Explanation)\b\s*[:\-]\s*(.+)")

    def __init__(self, model: LanguageModel, max_tokens: int, temperature: float, top_p: float):
        super().__init__(model, max_tokens, temperature, top_p)
        self._me2_bert_scorer = ME2BERTScorer()

    def generate_prompt(
        self,
        scenario: Scenario,
        question_format: str,
        distractor: Distractor | None,
    ) -> Prompt:
        question_template: QuestionTemplate = QUESTION_TEMPLATES[question_format]
        system_prompt = self._build_system_prompt(question_template["system"], distractor)
        user_prompt = question_template["user"].format(scenario["context"].strip())
        prompt: Prompt = {
            "id": f"{scenario["id"]}" +
                  f"_{distractor["id"] if distractor else "none"}",
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        return prompt

    def pre_process(
            self,
            scenario_series: pd.Series,
            question_format: str,
            distractor_series: pd.Series | None
    ) -> list[Prompt]:
        # Create Distractor
        distractor: Distractor | None = None
        if distractor_series is not None:
            modality = Modality(distractor_series["modality"])
            position = Position.BEFORE_SYSTEM
            distractor = {
                "id": str(distractor_series.get("id", distractor_series.get("distractor_id", ""))),
                "modality": modality,
                "file_path": str(distractor_series["file_path"]),
                "position": position
            }

        # Create Scenario
        scenario: Scenario = {
            "id": str(scenario_series.get("id", scenario_series.get("scenario_id", scenario_series.name))),
            "context": scenario_series.get("selftext", scenario_series.get("context", ""))
        }

        # Generate prompt
        prompt = self.generate_prompt(
            scenario=scenario,
            question_format=question_format,
            distractor=distractor
        )
        return [prompt]

    def _build_system_prompt(
        self,
        base_prompt: str,
        distractor: Distractor | None
    ) -> str:
        normalized_prompt = base_prompt.strip()
        if not distractor:
            return normalized_prompt

        if distractor["modality"] == Modality.TEXT:
            return self._prepend_text_distractor(normalized_prompt, distractor)

        if distractor["modality"] == Modality.IMAGE:
            return self._prepend_image_instruction(normalized_prompt)

        return normalized_prompt

    def _prepend_text_distractor(self, base_prompt: str, distractor: Distractor) -> str:
        distractor_path = PATH_DISTRACTORS / distractor["file_path"]
        with open(distractor_path, "r", encoding="utf-8") as f:
            distractor_text = f.read().strip()

        if not distractor_text:
            return base_prompt

        remainder = self._lowercase_first(base_prompt)
        return f"{distractor_text} Later, {remainder}" if remainder else distractor_text

    def _prepend_image_instruction(self, base_prompt: str) -> str:
        remainder = self._lowercase_first(base_prompt)
        if remainder:
            return f"You see the scene in the image. {remainder}"
        return "You see the scene in the image."

    @staticmethod
    def _lowercase_first(text: str) -> str:
        for idx, char in enumerate(text):
            if char.isalpha():
                return text[:idx] + char.lower() + text[idx + 1:]
        return text

    def post_process(
            self,
            prompt: Prompt,
            response: LanguageModelResponse
    ) -> dict[str, any]:
        distractor = prompt["distractor"]
        response_text = self._get_response_text(response)
        verdict, verdict_idx = self._extract_verdict(response_text)
        reasoning = self._extract_reasoning(response_text, verdict_idx)
        me2_result = self._me2_bert_scorer.predict([reasoning])[0]

        result = {
            "scenario_id": prompt["scenario"]["id"],
            "scenario_context": prompt["scenario"]["context"],
            "distractor_id": distractor["id"] if distractor else None,
            "distractor_modality": distractor["modality"].value if distractor else None,
            "distractor_file_path": distractor["file_path"] if distractor else None,
            "model_id": self.model.get_model_id(),
            "question_format": prompt.get("question_format", self._QUESTION_TYPE),
            "question_header": prompt["system_prompt"],
            "question_text": prompt["user_prompt"],
            "eval_technique": "top_p_sampling",
            "eval_top_p": self.top_p,
            "eval_temperature": self.temperature,
            "eval_max_tokens": self.max_tokens,
            "timestamp": response.timestamp,
            "answer_raw": response.answer_raw,
            "answer": response.answer,
            "response_text": response_text,
            "verdict": verdict,
            "reasoning": reasoning,
            "ch_score": me2_result.scores['CH'],
            "fc_score": me2_result.scores['FC'],
            "lb_score": me2_result.scores['LB'],
            "as_score": me2_result.scores['AS'],
            "pd_score": me2_result.scores['PD']
        }
        return result

    def _get_response_text(self, response: LanguageModelResponse) -> str:
        return (response.answer or response.answer_raw or "").strip()

    def _extract_verdict(self, response_text: str) -> tuple[str | None, int | None]:
        if not response_text:
            return None, None

        match = self._VERDICT_REGEX.search(response_text)
        if match:
            return match.group(1).upper(), match.end()

        for label in self._VALID_VERDICTS:
            fallback = re.search(rf"\b{label}\b", response_text, re.IGNORECASE)
            if fallback:
                return label, fallback.end()
        return None, None

    def _extract_reasoning(self, response_text: str, verdict_end: int | None) -> str | None:
        if not response_text:
            return None

        match = self._REASONING_REGEX.search(response_text)
        if match:
            snippet = match.group(1).strip()
            return self._strip_follow_up_headers(snippet)

        if verdict_end is not None and verdict_end < len(response_text):
            remainder = response_text[verdict_end:].strip()
            if remainder:
                return remainder
        return None

    def _strip_follow_up_headers(self, reasoning: str) -> str | None:
        cleaned = re.split(r"\n\s*(?:Verdict|Reasoning|Reason|Explanation|Summary)\s*[:\-]", reasoning, maxsplit=1)[0].strip()
        return cleaned or None
