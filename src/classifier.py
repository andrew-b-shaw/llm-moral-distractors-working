from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import PATH_HF_CACHE

logger = logging.getLogger(__name__)

# Keep HF downloads inside project cache
os.environ.setdefault("HF_HOME", str(PATH_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PATH_HF_CACHE))
# Optional: quiet the fork/parallelism warning if spawning processes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_ME2_KEYS: Sequence[str] = ("CH", "FC", "LB", "AS", "PD")
_FULL_NAMES: Dict[str, str] = {
    "CH": "CARE/HARM",
    "FC": "FAIRNESS/CHEATING",
    "LB": "LOYALTY/BETRAYAL",
    "AS": "AUTHORITY/SUBVERSION",
    "PD": "PURITY/DEGRADATION",
}

@dataclass
class ME2Result:
    text: str
    scores: Dict[str, float]  # {'CH': x, 'FC': y, 'LB': z, 'AS': a, 'PD': b}

class ME2BERTScorer:
    """
    Thin wrapper around lorenzozan/ME2-BERT.

    - Returns a dict for each input: {'CH','FC','LB','AS','PD'} -> float
    - No thresholds, no polarity heuristics.
    - Uses return_dict=True to follow the official model card.
    """

    def __init__(
        self,
        model_name: str = "lorenzozan/ME2-BERT",
        *,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 200,          # matches the card example
        trust_remote_code: bool = True, # required for custom model class
        revision: Optional[str] = None  # pin a known-good commit if you want strict reproducibility
    ) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self._batch_size = max(1, int(batch_size))

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Pin a revision for stability, remove `revision` if tracking latest.
        tok_kwargs = dict(trust_remote_code=trust_remote_code)
        mdl_kwargs = dict(trust_remote_code=trust_remote_code)
        if revision:
            tok_kwargs["revision"] = revision
            mdl_kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **mdl_kwargs).to(self.device)
        self.model.eval()

    def predict(self, texts: Sequence[str]) -> List[ME2Result]:
        if not texts:
            return []

        # Tokenize per model card, keep padding='max_length' so shapes are stable
        enc = self.tokenizer(
            list(texts),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        results: List[ME2Result] = []
        with torch.no_grad():
            # return_dict=True yields a list-like of dicts (one per input)
            # with keys 'CH','FC','LB','AS','PD'
            outputs = self.model(**enc, return_dict=True)

        # The model returns an indexable collection (len == batch size),
        # where each element acts like {'CH': float, ...}.
        for text, per_item in zip(texts, outputs):
            # Ensure stable key ordering and float-cast
            scores = {k: float(per_item[k]) for k in _ME2_KEYS if k in per_item}
            if len(scores) != 5:
                logger.warning("Unexpected ME2-BERT output keys: %s", list(per_item.keys()))
            results.append(ME2Result(text=text, scores=scores))
        return results


def _batch(items: Sequence[str], size: int):
    chunk: List[str] = []
    for it in items:
        chunk.append(it)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect ME²-BERT Moral Foundations activations.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=200, help="Token length (card examples use 200).")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto-detect by default).")
    parser.add_argument("--revision", default=None, help="HF commit hash to pin (e.g., e9940a6...).")
    parser.add_argument("texts", nargs="*", help="Texts to score; if omitted, a small demo set is used.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    examples = args.texts or [
        "I should tell the truth even if it hurts someone's feelings.",
        "Breaking the law to help a friend shows loyalty.",
        "I feel like the most moral thing to do in this situation is to ignore everything that the law says and just do what you think is the most fair for your friend.",
        "We will do a revolution like no other, and we should not trust the authority. We will break all of the laws.",
    ]

    clf = ME2BERTScorer(
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        revision=args.revision,  # e.g., "e9940a6c1aefb39b047033251486b1137f7da3e7"
    )
    preds = clf.predict(examples)

    for p in preds:
        print("\n", p.text)
        for k in _ME2_KEYS:
            print(f"  {k:2s} ({_FULL_NAMES[k]:>20s}): {p.scores.get(k, float('nan')):.5f}")