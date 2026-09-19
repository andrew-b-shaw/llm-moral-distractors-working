"""Pre-computes and caches result + significance-test data for every model/dataset
combination used across the analysis notebooks (moralchoice_bar_charts.ipynb,
result_and_sig_tables.ipynb), so those notebooks can load cached results instead of
re-running (expensive, crossed-random-effects) significance testing every time a
figure's appearance changes.

Safe to re-run: entries already cached, whose source CSVs haven't changed size or
mtime since, are skipped (see results_cache.load_or_compute). Re-run this after
adding new experiment CSVs, or after changing analysis logic in
moralchoice_analysis.py / normbank_analysis.py / reddit_analysis.py / significance_testing.py.

Usage: python -m src.analysis.compute_all_results
"""

import time

from src.analysis.moralchoice_analysis import calculate_moralchoice_results
from src.analysis.normbank_analysis import calculate_normbank_results
from src.analysis.reddit_analysis import calculate_reddit_verdict_results
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


def main():
    jobs = (
        [(calculate_moralchoice_results, args) for args in MORALCHOICE_CONFIGS] +
        [(calculate_normbank_results, args) for args in NORMBANK_CONFIGS] +
        [(calculate_reddit_verdict_results, args) for args in REDDIT_CONFIGS]
    )
    print(f"Computing/caching {len(jobs)} result sets (sig_testing=True)...", flush=True)
    for i, (fn, args) in enumerate(jobs, 1):
        t0 = time.time()
        load_or_compute(fn, *args, sig_testing=True)
        print(f"[{i}/{len(jobs)}] {fn.__name__}{args} -> {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
