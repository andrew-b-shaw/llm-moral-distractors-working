import pytest

from src.models.ollama_model import OllamaModel
from src.prompters.prompt import Modality, ImagePosition


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_ollama_query_builds_request_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"response": "Verdict: NTA"})

    monkeypatch.setattr("src.models.ollama_model.requests.post", fake_post)
    monkeypatch.setenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

    model = OllamaModel("ollama/gemma3-4b")
    prompt = {
        "scenario": {"id": "1", "context": "Example"},
        "distractor": {
            "id": "test",
            "modality": Modality.TEXT,
            "file_path": "unused.txt",
            "position": ImagePosition.BEFORE_SYSTEM
        },
        "system_prompt": "You are a helper.",
        "user_prompt": "Evaluate this situation."
    }
    response = model.query(
        prompt=prompt,
        max_tokens=64,
        temperature=0.4,
        top_p=0.8,
    )

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "gemma3:4b"
    assert "You are a helper." in captured["json"]["prompt"]
    assert "Evaluate this situation." in captured["json"]["prompt"]
    assert captured["json"]["options"]["num_predict"] == 64
    assert response.answer.strip().startswith("Verdict")


def test_ollama_query_raises_for_image_distractor(monkeypatch):
    model = OllamaModel("ollama/gemma3-4b")
    prompt = {
        "scenario": {"id": "1", "context": "context"},
        "distractor": {
            "id": "img",
            "modality": Modality.IMAGE,
            "file_path": "some.png",
            "position": ImagePosition.BEFORE_SYSTEM
        },
        "system_prompt": "System",
        "user_prompt": "User"
    }

    with pytest.raises(ValueError):
        model.query(prompt=prompt)
