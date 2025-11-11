from typing import TypedDict


class Distractor(TypedDict):
    modality: str  # "text" | "img"
    distractor_id: str
    text: str | None
    img_path: str | None