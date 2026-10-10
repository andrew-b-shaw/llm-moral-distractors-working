"""Pre-computes and caches result + significance-test data for every model/dataset
combination used across the analysis notebooks (moralchoice_bar_charts.ipynb,
normbank_bar_charts.ipynb, reddit_bar_charts.ipynb, result_and_sig_tables.ipynb),
so those notebooks can load cached results instead of re-running (expensive,
crossed-random-effects) significance testing every time a figure's appearance changes.

Safe to re-run: any config whose cache file already exists is skipped (see
results_cache.load_or_compute) regardless of whether the underlying CSV has changed
since -- there is no automatic staleness check, by design, since analyses are often
run on one machine and the cache carried over to another. If you've added new
experiment CSVs, changed analysis logic in moralchoice_analysis.py /
normbank_analysis.py / reddit_analysis.py / significance_testing.py, or just want a
config's numbers refreshed, delete its .pkl file(s) under PATH_ANALYSIS_RESULTS
yourself before re-running this script.

Usage: python -m src.analysis.compute_all_results
"""

import time

from src.analysis.moralchoice_analysis import calculate_moralchoice_results, disaggregate_moralchoice_results_from_files
from src.analysis.normbank_analysis import calculate_normbank_results, disaggregate_normbank_results_from_files
from src.analysis.reddit_analysis import (
    calculate_reddit_verdict_results, calculate_reddit_mf_results, disaggregate_reddit_verdict_results_from_files
)
from src.analysis.results_cache import load_or_compute

# every (csv, scenario) pair used by moralchoice_bar_charts.ipynb's result_configs
# and result_and_sig_tables.ipynb's moralchoice_{high,low}_ambiguity dataset_configs
MORALCHOICE_CONFIGS = [
    # main
    ("main/moralchoice_high_ambiguity/google_gemma-3-4b-it_moralchoice_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("main/moralchoice_high_ambiguity/meta-llama_Llama-3.2-3B-Instruct_moralchoice_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("main/moralchoice_high_ambiguity/Qwen_Qwen3-4B_moralchoice_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("main/moralchoice_high_ambiguity/openai_gpt-4.1_moralchoice_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("main/moralchoice_low_ambiguity/google_gemma-3-4b-it_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("main/moralchoice_low_ambiguity/meta-llama_Llama-3.2-3B-Instruct_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("main/moralchoice_low_ambiguity/Qwen_Qwen3-4B_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("main/moralchoice_low_ambiguity/Qwen_Qwen3-4b_moralchoice_low_ambiguity_short.csv", "moralchoice_low_ambiguity.csv"),
    ("main/moralchoice_low_ambiguity/openai_gpt-4.1_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    # multimodal
    ("multimodal/google_gemma-3-4b-it_multimodality_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("multimodal/google_gemma-3-4b-it_multimodality_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("multimodal/meta-llama_Llama-3.2-11B-Vision-Instruct_multimodality_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("multimodal/meta-llama_Llama-3.2-11B-Vision-Instruct_multimodality_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("multimodal/Qwen_Qwen3-VL-4B-Instruct_moralchoice_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("multimodal/Qwen_Qwen3-VL-4B-Instruct_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    # instruction-tuning ablation
    ("ablation/base/google_gemma-3-4b-pt_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/base/meta-llama_Llama-3.2-3B_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/base/Qwen_Qwen3-4B-Base_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    # prompt engineering ablation
    ("ablation/prompt engineering/google_gemma-3-4b-it_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/prompt engineering/meta-llama_Llama-3.2-3B-Instruct_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/prompt engineering/Qwen_Qwen3-4B_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/prompt engineering/openai_gpt-4.1_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    # size ablation
    ("ablation/size/google_gemma-3-27b-it_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/google_gemma-3-1b-it_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/google_gemma-3-270m-it_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/meta-llama_Llama-3.2-11B-Vision-Instruct_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/meta-llama_Llama-3.2-1B-Instruct_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/Qwen_Qwen3-32B_moralchoice_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/Qwen_Qwen3-1.7B_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    ("ablation/size/Qwen_Qwen3-0.6B_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
    # reasoning ablation
    ("ablation/thinking/Qwen_Qwen3-4B_thinking_high_ambiguity.csv", "moralchoice_high_ambiguity.csv"),
    ("ablation/thinking/Qwen_Qwen3-4B_thinking_low_ambiguity.csv", "moralchoice_low_ambiguity.csv"),
]

NORMBANK_CONFIGS = [
    ("main/normbank/google_gemma-3-4b-it_normbank.csv",),
    ("main/normbank/meta-llama_Llama-3.2-3B-Instruct_normbank.csv",),
    ("main/normbank/Qwen_Qwen3-4B_normbank.csv",),
    ("main/normbank/openai_gpt-4.1_normbank.csv",),
]

REDDIT_CONFIGS = [
    ("main/reddit/google_gemma-3-4b-it_reddit.csv",),
    ("main/reddit/meta-llama_Llama-3.2-3B-Instruct_reddit.csv",),
    ("main/reddit/Qwen_Qwen3-4B_reddit.csv",),
    ("main/reddit/openai_gpt-4.1_reddit.csv",),
]

# reddit_bar_charts.ipynb's judgement analysis marginalizes over all 5 verdict
# options (same call as REDDIT_CONFIGS above) and only graphs esh/yta/nta -- no
# separate cache entries needed, plot_multi_bar_chart's verdict_keys subsets at
# plot time rather than needing the underlying data recomputed with fewer options.

# reddit_bar_charts.ipynb's moral-foundations analysis, a different function entirely
REDDIT_MF_CONFIGS = [
    (args[0],) for args in REDDIT_CONFIGS
]

# distractor_disaggregation.ipynb's per-distractor-id breakdowns, each pooling all 4
# main models together. Note: these run one pairwise_mixed_lm fit per distractor_id
# (~30 per dataset) rather than per sentiment category (6), so they're much heavier
# than the configs above -- normbank/reddit additionally multiply that by their
# number of options.
# matches distractor_disaggregation.ipynb's dataset_configs exactly (the 4 main models)
DISAGG_MORALCHOICE_CONFIGS = [
    ([
        "main/moralchoice_high_ambiguity/google_gemma-3-4b-it_moralchoice_high_ambiguity.csv",
        "main/moralchoice_high_ambiguity/meta-llama_Llama-3.2-3B-Instruct_moralchoice_high_ambiguity.csv",
        "main/moralchoice_high_ambiguity/Qwen_Qwen3-4B_moralchoice_high_ambiguity.csv",
        "main/moralchoice_high_ambiguity/openai_gpt-4.1_moralchoice_high_ambiguity.csv",
    ], "moralchoice_high_ambiguity.csv"),
    ([
        "main/moralchoice_low_ambiguity/google_gemma-3-4b-it_moralchoice_low_ambiguity.csv",
        "main/moralchoice_low_ambiguity/meta-llama_Llama-3.2-3B-Instruct_moralchoice_low_ambiguity.csv",
        "main/moralchoice_low_ambiguity/Qwen_Qwen3-4B_moralchoice_low_ambiguity.csv",
        "main/moralchoice_low_ambiguity/openai_gpt-4.1_moralchoice_low_ambiguity.csv",
    ], "moralchoice_low_ambiguity.csv"),
]
DISAGG_NORMBANK_CONFIGS = [
    ([args[0] for args in NORMBANK_CONFIGS],),
]
DISAGG_REDDIT_CONFIGS = [
    ([args[0] for args in REDDIT_CONFIGS],),
]


def main():
    jobs = (
        # ablations skipped for now (main + visual only) -- restore MORALCHOICE_CONFIGS in
        # full to re-add instruction-tuning/prompt-engineering/size/reasoning ablations
        [(calculate_moralchoice_results, args, {}) for args in MORALCHOICE_CONFIGS[:15]] +
        [(calculate_normbank_results, args, {}) for args in NORMBANK_CONFIGS] +
        [(calculate_reddit_verdict_results, args, {}) for args in REDDIT_CONFIGS] +
        # reddit_mf skipped for now -- not needed by result_and_sig_tables.ipynb (only used
        # by reddit_bar_charts.ipynb's moral-foundations analysis), re-add when that's needed
        # [(calculate_reddit_mf_results, args, {}) for args in REDDIT_MF_CONFIGS] +
        [(disaggregate_moralchoice_results_from_files, args, {}) for args in DISAGG_MORALCHOICE_CONFIGS] +
        [(disaggregate_normbank_results_from_files, args, {}) for args in DISAGG_NORMBANK_CONFIGS] +
        [(disaggregate_reddit_verdict_results_from_files, args, {}) for args in DISAGG_REDDIT_CONFIGS]
    )
    print(f"Computing/caching {len(jobs)} result sets (sig_testing=True)...", flush=True)
    for i, (fn, args, kwargs) in enumerate(jobs, 1):
        t0 = time.time()
        load_or_compute(fn, *args, sig_testing=True, **kwargs)
        print(f"[{i}/{len(jobs)}] {fn.__name__}{args} {kwargs} -> {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
