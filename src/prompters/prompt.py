from enum import StrEnum
from typing import TypedDict


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class Distractor(TypedDict):
    id: str
    modality: Modality
    text: str | None
    image_path: str | None


class Scenario(TypedDict):
    id: str
    context: str


class Prompt(TypedDict):
    scenario: Scenario
    distractor: Distractor | None
    system_prompt: str
    user_prompt: str