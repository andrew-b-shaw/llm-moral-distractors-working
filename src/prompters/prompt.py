"""Core data structures for the prompting pipeline.

Defines Prompt (the unit passed to a model's query method), Scenario, Distractor,
and enums for distractor modality (text vs. image) and image position within
the message sequence."""

from enum import Enum
from typing import TypedDict, Optional

try:  # pragma: no cover
    from enum import StrEnum  # type: ignore
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        """Fallback for Python < 3.11."""
        pass


class ImagePosition(StrEnum):
    BEFORE_SYSTEM = "before_system"
    AFTER_SYSTEM = "after_system"
    BEFORE_USER = "before_user"
    AFTER_USER = "after_user"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class Distractor(TypedDict):
    id: str
    modality: Modality
    file_path: str
    position: ImagePosition


class Scenario(TypedDict):
    id: str
    context: str


class Prompt(TypedDict):
    id: str
    scenario: Scenario
    distractor: Optional[Distractor]
    system_prompt: str
    user_prompt: str
