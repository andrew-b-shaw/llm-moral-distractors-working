from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from src.models.model_utils import get_timestamp
from src.models.model import MODELS, LanguageModel, LanguageModelResponse
from src.prompters.prompt import Modality, Prompt


class OllamaModelResponse(LanguageModelResponse):
    def get_answer_prob(self, answer: str) -> float:
        # Ollama HTTP API does not expose token probabilities.
        return 0.0


class OllamaModel(LanguageModel):
    """Thin wrapper around a locally running Ollama server."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        assert MODELS[model_name]["model_class"] == "OllamaModel", (
            f"Erroneous Model Instantiation for {model_name}"
        )
        self._ollama_model = MODELS[model_name]["ollama_model"]
        self._api_url = os.environ.get(
            "OLLAMA_API_URL",
            "http://localhost:11434/api/generate"
        )
        self._timeout = float(os.environ.get("OLLAMA_API_TIMEOUT", "120"))

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        system = system_prompt.strip()
        user = user_prompt.strip()
        if system and user:
            return f"{system}\n\n{user}"
        return system or user

    def query(
        self,
        prompt: Prompt,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> OllamaModelResponse:
        distractor = prompt.get("distractor")
        if distractor and distractor["modality"] == Modality.IMAGE:
            raise ValueError("Ollama text endpoint does not support image distractors.")

        text_prompt = self._build_prompt(prompt["system_prompt"], prompt["user_prompt"])

        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "prompt": text_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }
        response = requests.post(
            self._api_url,
            json=payload,
            timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()
        answer_raw = data.get("response", "")
        answer = answer_raw.strip()
        return OllamaModelResponse(
            timestamp=get_timestamp(),
            answer_raw=answer_raw,
            answer=answer
        )
