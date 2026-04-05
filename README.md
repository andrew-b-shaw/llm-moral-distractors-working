# Are Language Models Sensitive to Morally Irrelevant Distractors?

Repository for the project *"Are Language Models Sensitive to Morally Irrelevant Distractors?"* (Andrew Shaw, Christina Hahn, Catherine Rasgaitis, Yash Mishra, Alisa Liu, Natasha Jaques, Yulia Tsvetkov, Amy X. Zhang).

**Project link:** [https://github.com/andrew-b-shaw/llm-moral-distractors](https://github.com/andrew-b-shaw/llm-moral-distractors)

## What this repository contains

This codebase runs **moral benchmark evaluations** with optional **distractors** (emotionally valenced text or images). Distractor metadata and paths are in `data/distractors/distractors.csv`.

Three evaluation tracks are registered in `data/scenarios/dataset_configs.py`:

| Track | Data source | What the code does |
| ----- | ----------- | ------------------- |
| **MoralChoice** | `data/scenarios/moralchoice_low_ambiguity.csv`, `moralchoice_high_ambiguity.csv` | A/B-style forced choice via `MoralChoicePrompter`. |
| **Norm Bank** | `data/scenarios/normbank.csv` | Good / acceptable / wrong via `NormBankPrompter`. |
| **Reddit (r/AITA-style)** | Hugging Face dataset named in the `reddit` entry of `dataset_configs.py`; file loaded via `hf_hub_download` in `evaluate.py` | Verdict + reasoning via `RedditPrompter`. Cap with `--eval-num-scenarios`. |

Documentation here describes the repository files, CLI, and configuration only.

## Acknowledgments

The evaluation design (MoralChoice-style prompting, token-level answer scoring from Hugging Face generation logits and from OpenAI logprobs where used, template structure) adapts the open-source [MoralChoice codebase from Scherrer et al.](https://github.com/ninodimontalcino/moralchoice). Original code is MIT-licensed; see [LICENSE.md](LICENSE.md).

## Repository structure

```
├── src/
│   ├── evaluate.py              # CLI: load scenarios → query model → write pickles under data/responses/
│   ├── collect.py               # Walk pickles → aggregate CSVs under data/csv_results/
│   ├── config.py                # Root-relative paths
│   ├── classifier.py            # ME2-BERT wrapper (Reddit prompter)
│   ├── models/
│   │   ├── model.py             # LanguageModel / LanguageModelResponse abstractions
│   │   ├── model_configs.py     # Supported --model-name registry
│   │   ├── model_utils.py       # API keys, timestamps
│   │   ├── openai_model.py      # OpenAI Chat Completions (+ batch helpers)
│   │   ├── llama_model.py       # Hugging Face Llama 3.2
│   │   ├── gemma_model.py       # Hugging Face Gemma 3 (text + vision-capable variants)
│   │   ├── qwen_model.py        # Hugging Face Qwen 3 (text)
│   │   └── qwen_vl_model.py     # Hugging Face Qwen 3 VL
│   ├── prompters/               # moralchoice_prompter, normbank_prompter, reddit_prompter, …
│   ├── analysis/                # Notebooks (not covered in detail here)
│   └── test/
├── data/
│   ├── scenarios/               # Scenario CSVs + dataset_configs.py
│   ├── distractors/             # distractors.csv + text/image assets
│   └── templates/               # question_templates.py, response_templates.py
├── api_keys/                    # Expected key files (often gitignored)
├── cache/                       # HF cache default (see config.py)
└── requirements.txt
```

## Setup

### Python

Use an environment compatible with `requirements.txt`. This project was developed with **Python 3.12.12**.

```bash
pip install -r requirements.txt
```

`requirements.txt` reflects the experiment environment and includes pinned CUDA packages (`nvidia-*`). In practice, the full install is intended for a **Linux/CUDA** machine when you plan to run the local Hugging Face backends.

### API keys

`get_api_key()` in `src/models/model_utils.py` reads:

- `api_keys/openai_key.txt` — OpenAI API
- `api_keys/huggingface_key.txt` — required for local Hugging Face models (`Llama`, `Gemma`, `Qwen`, `Qwen-VL`)

### GPU

Local backends under `src/models/` (Gemma, Llama, Qwen, Qwen-VL) expect a **CUDA** setup matching your installed PyTorch build.

## Usage

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
  --experiment-name "my_experiment" \
  --dataset "moralchoice_low_ambiguity" \
  --model-name "google/gemma-3-4b-it" \
  --distractors "text" \
  --question-formats "ab" \
  --eval-max-tokens 3
```

Common flags (full list in `src/evaluate.py`):

- `--dataset`: `moralchoice_high_ambiguity` | `moralchoice_low_ambiguity` | `normbank` | `reddit`
- `--model-name`: must appear in `src/models/model_configs.py`
- `--distractors`: `text` | `image` | `all` | `none`
- `--question-formats`: e.g. `ab` for MoralChoice (templates append `_moralchoice` internally)

### Collecting CSVs

```bash
python -m src.collect \
  --experiment-name "my_experiment" \
  --dataset "moralchoice_low_ambiguity" \
  --distractors "text"
```

`collect.py` keeps pickles whose **filenames** contain `txt`, `img`, or `none` (baseline uses distractor id `none`). Matching works with current distractor ids (`txt_*`, `img_*`). Use `all` to load every `.pickle` found. Use the **same** `--dataset` value as in `evaluate.py`.

### OpenAI Batch API

Use `--batch-submit` / `--batch-retrieve` together with `--batch-submit-output-filename` and `--batch-retrieve-response-filename` as enforced by `evaluate.py`. See `src/run_openai_batch.ipynb`.
