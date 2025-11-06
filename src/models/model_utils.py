import os
from datetime import datetime

from src.config import PATH_API_KEYS


def get_api_key(company_identifier: str) -> str:
    """
    Helper Function to retrieve API key from files
    """
    path_key = str(PATH_API_KEYS / f"{company_identifier}_key.txt")

    if os.path.exists(path_key):
        with open(path_key, encoding="utf-8") as f:
            key = f.read()
        return key

    raise ValueError(f"API KEY not available at: {path_key}")


def get_raw_likelihoods_from_answer(
        token_likelihoods: dict[str, float], start: int = 0, end: int = None
) -> list[float]:
    """
    Helper Function to filter token_likelihood
    """
    if token_likelihoods[start][0] in [":", "\s", ""]:
        start += 1

    likelihoods = [
        likelihood
        for (token, likelihood) in token_likelihoods[start:end]
        if token not in ["<BOS_TOKEN>", "</s>"]
    ]

    return likelihoods


def get_timestamp():
    """
    Generate timestamp of format Y-M-D_H:M:S
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")