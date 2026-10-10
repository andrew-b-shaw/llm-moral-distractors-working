"""Disk cache for the (expensive, mixed-effects-significance-testing-heavy) outputs of
calculate_moralchoice_results / calculate_normbank_results / calculate_reddit_verdict_results /
calculate_reddit_mf_results / disaggregate_*_results_from_files.

Each distinct call is cached to its own pickle file under PATH_ANALYSIS_RESULTS, named after
the experiment (function + arguments) so the directory stays easy to browse. There is no
freshness check: analyses are often run on one machine and the cache carried over to another
(file mtimes aren't portable across a git pull/clone/transfer, so a mtime-based check couldn't
work here anyway), so a cache entry is used as-is whenever its file exists. If you've changed
the underlying data (new experiment CSVs, or a code change to how a result is computed) and
want it recomputed, delete the relevant .pkl file(s) yourself -- see PATH_ANALYSIS_RESULTS.
"""

import pickle
import re

from src.config import PATH_ANALYSIS_RESULTS


def _readable(value):
    if isinstance(value, (list, tuple)):
        return "_".join(_readable(v) for v in value)
    return str(value)


def _cache_name(fn, args, kwargs):
    parts = [fn.__name__, *(_readable(a) for a in args)]
    parts += [f"{k}={v}" for k, v in sorted(kwargs.items()) if k != "sig_testing"]
    readable = re.sub(r'[^A-Za-z0-9]+', '_', "__".join(parts)).strip('_')
    return readable[:180]


def load_or_compute(fn, *args, **kwargs):
    """Return fn(*args, **kwargs), loading from the on-disk cache if that experiment's file
    already exists. No staleness check -- see module docstring."""
    PATH_ANALYSIS_RESULTS.mkdir(parents=True, exist_ok=True)
    cache_file = PATH_ANALYSIS_RESULTS / f"{_cache_name(fn, args, kwargs)}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    result = fn(*args, **kwargs)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    return result
