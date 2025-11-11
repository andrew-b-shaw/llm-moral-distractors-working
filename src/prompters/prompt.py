from enum import StrEnum
from typing import TypedDict


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class Distractor(TypedDict):
    id: str
    modality: Modality
    file_path: str


class Scenario(TypedDict):
    id: str
    context: str


class Prompt(TypedDict):
    scenario: Scenario
    distractor: Distractor | None
    system_prompt: str
    user_prompt: str