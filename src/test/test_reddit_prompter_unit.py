import datetime as dt

import pandas as pd

from src.prompters.reddit_prompter import RedditPrompter
from src.models.models import LanguageModelResponse


class DummyResponse(LanguageModelResponse):
    def __init__(self, answer_text: str):
        super().__init__(
            timestamp=dt.datetime.utcnow().isoformat(),
            answer_raw=answer_text,
            answer=answer_text,
        )

    def get_answer_prob(self, answer: str) -> float:  # pragma: no cover - not used
        return 0.0


class DummyModel:
    def __init__(self, responses: list[LanguageModelResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)

    def get_model_id(self):
        return "dummy/model"


def build_prompter(response_text: str = "Verdict: NTA\nReasoning: ok") -> RedditPrompter:
    model = DummyModel([DummyResponse(response_text)])
    return RedditPrompter(
        model=model,
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )


def test_pre_process_builds_prompt_without_distractor():
    prompter = build_prompter()
    scenario = pd.Series({"id": "foo", "selftext": "Example reddit post"})

    prompts = prompter.pre_process(scenario, "reddit", distractor_series=None)

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt["scenario"]["id"] == "foo"
    assert "Example reddit post" in prompt["user_prompt"]
    assert prompt["distractor"] is None


def test_pre_process_includes_optional_distractor():
    prompter = build_prompter()
    scenario = pd.Series({"id": "foo", "selftext": "Example reddit post"})
    distractor = pd.Series({"id": "10", "modality": "text", "file_path": "file.txt"})

    prompts = prompter.pre_process(scenario, "reddit", distractor_series=distractor)

    distractor_dict = prompts[0]["distractor"]
    assert distractor_dict["id"] == "10"
    assert distractor_dict["modality"].value == "text"
    assert distractor_dict["file_path"] == "file.txt"


def test_prompt_invokes_model_and_post_processes_result():
    answer_text = "Verdict: YTA\nReasoning: You ignored everyone."
    prompter = build_prompter(response_text=answer_text)
    scenario = pd.Series({"id": "foo", "selftext": "Example reddit post"})

    results = prompter.prompt("reddit", scenario_series=scenario)

    assert len(results) == 1
    assert results[0]["verdict"] == "YTA"
    assert "ignored everyone" in results[0]["reasoning"]
    assert prompter.model.get_model_id() in results[0]["model_id"]
