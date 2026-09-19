"""Disk cache for the (expensive, mixed-effects-significance-testing-heavy) outputs of
calculate_moralchoice_results / calculate_normbank_results / calculate_reddit_verdict_results.

Each distinct call is cached to its own pickle file under PATH_ANALYSIS_RESULTS, so that
notebooks can reload precomputed results (e.g. after tweaking a plot's appearance) without
re-running significance testing. A cache entry is invalidated automatically if the size or
modification time of any underlying CSV/scenario/distractor file the call reads has changed.
"""

import hashlib
import pickle
import re

from src.config import PATH_ANALYSIS_RESULTS, PATH_CSV_RESULTS, PATH_SCENARIOS, PATH_DISTRACTORS

_PATH_BASES = [PATH_CSV_RESULTS, PATH_SCENARIOS, PATH_DISTRACTORS]


def _file_signature(value):
    """(size, mtime) for `value` if it resolves to a real file under a known data dir, else None."""
    if not isinstance(value, str):
        return None
    for base in _PATH_BASES:
        candidate = base / value
        if candidate.is_file():
            stat = candidate.stat()
            return stat.st_size, stat.st_mtime
    return None


def _cache_key(fn, args, kwargs):
    readable = re.sub(r'[^A-Za-z0-9]+', '_', str(args[0]) if args else fn.__name__).strip('_')[:80]
    signatures = [_file_signature(v) for v in (*args, *kwargs.values())]
    digest_src = repr((fn.__name__, args, sorted(kwargs.items()), signatures))
    digest = hashlib.md5(digest_src.encode()).hexdigest()[:10]
    return f"{fn.__name__}__{readable}__{digest}"


def load_or_compute(fn, *args, **kwargs):
    """Return fn(*args, **kwargs), from the on-disk cache if a fresh entry exists."""
    PATH_ANALYSIS_RESULTS.mkdir(parents=True, exist_ok=True)
    cache_file = PATH_ANALYSIS_RESULTS / f"{_cache_key(fn, args, kwargs)}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    result = fn(*args, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    return result
