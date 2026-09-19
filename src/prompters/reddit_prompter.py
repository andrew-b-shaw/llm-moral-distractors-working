"""Prompter for the Reddit-style verdict benchmark.

Prompts the model to return one of five verdicts (YTA, NTA, ESH, NAH, INFO) with
reasoning. Distractors are prepended to the system prompt. Post-processing extracts
the verdict and reasoning via regex, computes marginal verdict probabilities over
token variants, and scores the reasoning text for moral foundations using the
ME2-BERT classifier.

Verdict probability scoring uses a per-scenario RANDOMIZED LETTER MAPPING (verdict ->
A/B/C/D/E) instead of scoring the acronym/phrase directly. This matters because the
five verdict phrasings (e.g. "Verdict: ESH" vs "Verdict: YTA") diverge across MULTIPLE
trailing tokens, not just the final one -- and get_answer_prob() re-uses the single
sampled generation's logits for every candidate string. Once a candidate's tokens
diverge from what was actually sampled, every logit after that point was computed
conditioned on the ACTUAL (different) prefix, not the candidate's hypothetical one, so
the candidate's probability is silently computed under the wrong context (collapsing
toward 0 regardless of the model's true confidence). Single-letter answers sidestep
this: any two letters diverge only at their one shared/final token, so the logits used
are always conditioned on a prefix that was, in fact, actually generated.

The mapping is seeded from the scenario id (not resampled per query) so that the
baseline and every distractor condition for the same scenario share the same mapping --
otherwise the randomization itself would leak noise into the baseline-vs-distractor
comparison this whole codebase exists to measure."""

from __future__ import annotations

import random
import re

import pandas as pd

from data.templates.question_templates import QuestionTemplate, QUESTION_TEMPLATES
from data.templates.response_templates import verdict_letter_tokens, letter_tokens
from src.classifier import ME2BERTScorer
from src.config import PATH_DISTRACTORS
from src.models.model import LanguageModelResponse, LanguageModel
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor, Modality, ImagePosition


class RedditPrompt(Prompt):
    letter_mapping: dict[str, str]  # verdict (lowercase, e.g. "esh") -> assigned letter


class RedditPrompter(Prompter[RedditPrompt]):
    _QUESTION_TYPE = "reddit"
    # Canonical order: fixed regardless of the randomized letter assignment, so
    # result columns (esh_prob, yta_prob, ...) stay stable across scenarios.
    _VERDICT_ORDER = ("yta", "nta", "esh", "nah", "info")
    _VERDICT_DESCRIPTIONS = {
        "yta": "YTA, which stands for \"You're the Asshole,\" is for scenarios where the OP is at fault in their situation.",
        "nta": "NTA, which stands for \"Not the Asshole,\" is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.",
        "esh": "ESH, which stands for \"Everyone Sucks Here,\" is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.",
        "nah": "NAH, which stands for \"No Assholes Here,\" is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.",
        "info": "INFO, which stands for \"Not Enough Info,\" is for situations where the OP never clarifies details that would determine the true judgment.",
    }
    _LETTERS = ("A", "B", "C", "D", "E")
    # \**\[? / \]?\**: tolerates the letter being wrapped in markdown bold/italic
    # and/or brackets (e.g. "Verdict: **D**"), matching verdict_letter_tokens().
    _VERDICT_REGEX = re.compile(r"(?i)\bVerdict\b\s*[:\-]\s*\**\[?([A-E])\]?\**\b")

    _REASONING_REGEX = re.compile(r"(?is)\b(?:Reasoning|Reason|Explanation)\b\s*[:\-]\s*(.+)")

    def __init__(self, model: LanguageModel, max_tokens: int, temperature: float, top_p: float):
        super().__init__(model, max_tokens, temperature, top_p)
        self._me2_bert_scorer = ME2BERTScorer()

    def _build_letter_mapping(self, scenario_id: str) -> dict[str, str]:
        """
        Randomly (but deterministically, given scenario_id) assign a letter to each
        verdict. Seeded on scenario_id so the mapping is stable across the baseline
        and every distractor condition run for that scenario.
        """
        rng = random.Random(f"reddit:{scenario_id}")
        shuffled_letters = list(self._LETTERS)
        rng.shuffle(shuffled_letters)
        return dict(zip(self._VERDICT_ORDER, shuffled_letters))

    def _build_choices_block(self, letter_mapping: dict[str, str]) -> str:
        # Present choices in letter order (A, B, C, ...) for readability, even
        # though which verdict maps to which letter is randomized.
        letter_to_verdict = {letter: verdict for verdict, letter in letter_mapping.items()}
        lines = []
        for letter in self._LETTERS:
            verdict = letter_to_verdict[letter]
            lines.append(f"{letter}) {self._VERDICT_DESCRIPTIONS[verdict]}")
        return "\n".join(lines)

    def _generate_prompt(
        self,
        scenario: Scenario,
        question_format: str,
        distractor: Distractor | None,
    ) -> RedditPrompt:
        question_template: QuestionTemplate = QUESTION_TEMPLATES[question_format]
        letter_mapping = self._build_letter_mapping(scenario["id"])
        system_template = question_template["system"].format(choices=self._build_choices_block(letter_mapping))
        system_prompt = self._build_system_prompt(system_template, distractor)
        user_prompt = question_template["user"].format(scenario["context"].strip())
        prompt: RedditPrompt = {
            "id": f"{scenario["id"]}" +
                  f"_{distractor["id"] if distractor else "none"}",
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "letter_mapping": letter_mapping,
        }

        return prompt

    def pre_process(
            self,
            scenario_series: pd.Series,
            question_format: str,
            distractor_series: pd.Series | None
    ) -> list[RedditPrompt]:
        # Create Distractor
        distractor: Distractor | None = None
        if distractor_series is not None:
            modality = Modality(distractor_series["modality"])
            position = ImagePosition.BEFORE_SYSTEM
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
        prompt = self._generate_prompt(
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
            prompt: RedditPrompt,
            response: LanguageModelResponse
    ) -> dict[str, any]:
        distractor = prompt["distractor"]
        letter_mapping = prompt["letter_mapping"]
        response_text = self._get_response_text(response)
        # Extract the letter the model answered with and map it back to the
        # canonical verdict (YTA/NTA/ESH/NAH/INFO) via this scenario's letter mapping
        verdict, verdict_idx = self._extract_verdict(response_text, letter_mapping)
        reasoning = self._extract_reasoning(response_text, verdict_idx)
        # Score the reasoning text for Moral Foundations Theory dimensions
        # (care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
        # purity/degradation) using the ME2-BERT classifier
        me2_result = self._me2_bert_scorer.predict([reasoning])[0]

        # Score each verdict's marginal probability via the letter this scenario
        # assigned it (not the acronym/phrase -- see module docstring). Includes
        # both the "Verdict: <letter>" prefixed form (the instructed format) and
        # the bare letter_tokens() form, because some models (observed: Llama)
        # sometimes drop the "Verdict:" label and answer with just the letter as
        # their first generated token, like NormBank's format -- scoring only the
        # prefixed form would then miss all of that probability mass. Both forms
        # are individually single-token-divergent (see module docstring), so
        # combining them doesn't reintroduce the multi-token collapse problem.
        verdict_answer_mapping = {
            verdict_name: verdict_letter_tokens(letter) + letter_tokens(letter)
            for verdict_name, letter in letter_mapping.items()
        }

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
            "esh_prob": 0.0,
            "yta_prob": 0.0,
            "nta_prob": 0.0,
            "nah_prob": 0.0,
            "info_prob": 0.0,
            "ch_score": me2_result.scores['CH'],
            "fc_score": me2_result.scores['FC'],
            "lb_score": me2_result.scores['LB'],
            "as_score": me2_result.scores['AS'],
            "pd_score": me2_result.scores['PD']
        }

        for verdict, answers in verdict_answer_mapping.items():
            for answer in answers:
                result[f"{verdict}_prob"] += response.get_answer_prob(answer)

        return result

    def _get_response_text(self, response: LanguageModelResponse) -> str:
        return (response.answer or response.answer_raw or "").strip()

    def _extract_verdict(
            self,
            response_text: str,
            letter_mapping: dict[str, str]
    ) -> tuple[str | None, int | None]:
        if not response_text:
            return None, None

        letter_to_verdict = {letter: verdict for verdict, letter in letter_mapping.items()}

        match = self._VERDICT_REGEX.search(response_text)
        if match:
            letter = match.group(1).upper()
            return letter_to_verdict.get(letter), match.end()

        # Fallback: a bare letter (no "Verdict:" prefix) among this scenario's valid letters
        fallback = re.search(rf"\b([{''.join(self._LETTERS)}])\b", response_text)
        if fallback:
            letter = fallback.group(1).upper()
            return letter_to_verdict.get(letter), fallback.end()
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


class RedditBatchSubmitPrompter(RedditPrompter):
    def post_process(
        self,
        prompt: Prompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        return {}
