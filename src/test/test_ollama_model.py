from pathlib import Path
from unittest import mock

import pytest

from src.models.ollama_model import OllamaModel
from src.prompters.prompt import Modality


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_ollama_query_builds_request_payload(monkeypatch, tmp_path):
    temp_file = Path("data")
    temp_file.mkdir(exist_ok=True)
    distractor_path = temp_file / "tmp_ollama_test.txt"
    distractor_path.write_text("Remember the rules.")

    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"response": "Verdict: NTA"})

    monkeypatch.setattr("src.models.ollama_model.requests.post", fake_post)
    monkeypatch.setenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

    model = OllamaModel("ollama/gemma3-4b")
    response = model.query(
        user_prompt="Evaluate this situation.",
        system_prompt="You are a helper.",
        max_tokens=64,
        temperature=0.4,
        top_p=0.8,
        distractor={
            "id": "test",
            "modality": Modality.TEXT,
            "file_path": "tmp_ollama_test.txt"
        }
    )

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "gemma3:4b"
    assert "Remember the rules." in captured["json"]["prompt"]
    assert captured["json"]["options"]["num_predict"] == 64
    assert response.answer.strip().startswith("Verdict")
    distractor_path.unlink()
