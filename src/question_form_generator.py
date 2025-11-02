""" Question Form Generator"""
import json
from typing import Dict, Tuple

from data.question_templates.question_templates import QUESTION_TEMPLATES
from src.config import PATH_QUESTION_TEMPLATES


def get_question_form(
    scenario: Dict, question_type: str, question_ordering: int, system_instruction: bool
) -> Tuple[Dict, Dict]:
    """Get question form for a given scenario, question_type and question_ordering"""

    # (0) Set option ordering
    action_mapping = (
        {"A": "action1", "B": "action2"}
        if question_ordering == 0
        else {"A": "action2", "B": "action1"}
    )

    # (2) Generate question form
    question_config = QUESTION_TEMPLATES[question_type]

    question_form = {
        "question": question_config["question"].format(
            scenario["selftext"]
        )
        if question_type == "aita" else
        question_config["question"].format(
            scenario["context"],
            scenario[action_mapping["A"]],
            scenario[action_mapping["B"]],
        ),
        "question_header": question_config["question_header"]
        if system_instruction
        else "",
    }

    return (question_form, action_mapping)
