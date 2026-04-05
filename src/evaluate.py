"""
Main CLI entry point for running moral distractor experiments.

Loads scenarios from a moral benchmark dataset (MoralChoice, Norm Bank, or r/AITA),
optionally pairs them with emotionally-valenced distractors, queries a language model,
and writes per-scenario results as pickle files under data/responses/.
"""

import itertools
import os
import pickle
import argparse
import sys
import gc
from pathlib import Path
from typing import Optional

from src.models.model_configs import MODELS

os.environ.setdefault("HF_DATASETS_USE_DILL", "0")

import torch
import pandas as pd
from tqdm import tqdm
from datasets import config as datasets_config
from huggingface_hub import hf_hub_download

from src.prompters.moralchoice_prompter import MoralChoicePrompter, MoralChoiceBatchSubmitPrompter
from src.prompters.reddit_prompter import RedditPrompter, RedditBatchSubmitPrompter
from src.prompters.normbank_prompter import NormBankPrompter, NormBankBatchSubmitPrompter

from src.models.model import BatchSubmitLanguageModel, BatchRetrieveLanguageModel
from src.models.openai_model import OpenAIModel, OpenAIBatchSubmitModel, OpenAIBatchRetrieveModel
from src.models.gemma_model import GemmaModel
from src.models.llama_model import LlamaModel
from src.models.qwen_model import QwenModel
from src.models.qwen_vl_model import QwenVLModel

from src.config import PATH_DISTRACTORS, PATH_RESULTS, PATH_HF_CACHE
from data.scenarios.dataset_configs import DATASETS

################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="LLM Ethics Benchmark Evaluation with Moral Distractors")
parser.add_argument(
    "--experiment-name",
    required=True,
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset",
    required=True,
    type=str,
    help="Dataset to evaluate (moralchoice_high_ambiguity, moralchoice_low_ambiguity, normbank, reddit)",
)
parser.add_argument(
    "--distractors",
    default="none",
    type=str,
    help="Which distractors to use (all, text, image, none)",
)
parser.add_argument(
    "--batch-submit",
    action='store_true',
    default=False,
    help="Whether to create batch request for submission",
)
parser.add_argument(
    "--batch-retrieve",
    action='store_true',
    default=False,
    help="Whether to retrieve batch request",
)
parser.add_argument(
    "--model-name",
    required=True,
    type=str,
    help="Model to evaluate — see src/models/model_configs.py for supported identifiers",
)
parser.add_argument(
    "--question-formats",
    default=None,
    type=str,
    help="Question Templates to evaluate (defaults depend on dataset)",
    nargs="+",
)
parser.add_argument(
    "--eval-top-p",
    default=None,
    type=float,
    help="Top-P parameter for sampling (defaults to dataset-specific value)",
)
parser.add_argument(
    "--eval-temp",
    default=None,
    type=float,
    help="Temperature for sampling (defaults to dataset-specific value)",
)
parser.add_argument(
    "--eval-max-tokens",
    default=200,
    type=int,
    help="Max. number of tokens per completion",
)
parser.add_argument(
    "--eval-num-samples",
    default=1,
    type=int,
    help="Num. of samples per question form"
)
parser.add_argument(
    "--eval-num-scenarios",
    default=-1,
    type=int,
    help="Num. of scenarios to evaluate"
)
parser.add_argument(
    "--reddit-dataset-name",
    default="ucberkeley-dlab/normative_evaluation_llms_everyday_dilemmas",
    type=str,
    help="Hugging Face dataset identifier to use for the reddit configuration.",
)
parser.add_argument(
    "--reddit-text-column",
    default="selftext",
    type=str,
    help="Column in the reddit dataset that contains the scenario text/selftext.",
)
parser.add_argument(
    "--reddit-id-column",
    default="submission_id",
    type=str,
    help="Column in the reddit dataset that should be treated as the scenario id.",
)
parser.add_argument(
    "--batch-submit-output-filename",
    default=None,
    type=str,
    help="Name of file to output requests to for batch submission.",
)
parser.add_argument(
    "--batch-retrieve-response-filename",
    default=None,
    type=str,
    help="Name of file with responses for batch retrieval.",
)

args = parser.parse_args()
datasets_config.USE_DILL_FOR_PICKLING = False

################################################################################################
# HELPER METHODS
################################################################################################

def _get_dataset_config(dataset_name: str) -> dict:
    try:
        return DATASETS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset_name}'") from exc

def _load_csv_scenarios(path: Path, max_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    scenarios = pd.read_csv(path)
    if max_rows > 0:
        scenarios = scenarios.iloc[:max_rows]
    return scenarios

def _load_reddit_scenarios(max_rows: int) -> pd.DataFrame:
    dataset_name = args.reddit_dataset_name or DATASETS["reddit"]["hf_dataset_name"]
    dataset_config = _get_dataset_config("reddit")
    dataset_file = dataset_config.get("hf_dataset_file", "cleaned_dataset.csv")
    text_column = args.reddit_text_column
    id_column = args.reddit_id_column

    cache_path = hf_hub_download(
        repo_id=dataset_name,
        filename=dataset_file,
        repo_type="dataset",
        cache_dir=str(PATH_HF_CACHE),
    )

    read_kwargs = {
        "usecols": lambda col: col in {text_column, id_column},
    }
    if max_rows > 0:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(cache_path, **read_kwargs)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in reddit dataset file '{dataset_file}'."
        )

    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]

    if id_column in df.columns:
        df["id"] = df[id_column].astype(str)
    else:
        df["id"] = df.index.astype(str)

    scenarios = df.rename(columns={text_column: "selftext"})[["id", "selftext"]]
    scenarios.reset_index(drop=True, inplace=True)
    return scenarios

def load_scenarios(dataset_name: str, max_rows: int) -> pd.DataFrame:
    dataset_config = _get_dataset_config(dataset_name)
    loader_key = dataset_config.get("scenario_loader", "csv")
    if loader_key == "csv":
        return _load_csv_scenarios(dataset_config["scenario_path"], max_rows)
    if loader_key == "reddit":
        return _load_reddit_scenarios(max_rows)
    raise ValueError(f"Unsupported loader '{loader_key}' for dataset '{dataset_name}'.")

def load_distractors(setting: str) -> Optional[pd.DataFrame]:
    setting = (setting or "none").lower()
    if setting == "none":
        return None

    distractor_path = PATH_DISTRACTORS / "distractors.csv"
    if not distractor_path.exists():
        raise FileNotFoundError(f"Distractor file not found: {distractor_path}")

    distractors = pd.read_csv(distractor_path)
    if setting == "text":
        distractors = distractors[distractors["modality"] == "text"]
    elif setting == "image":
        distractors = distractors[distractors["modality"] == "image"]
    elif setting not in {"all"}:
        raise ValueError(
            f"Unknown distractor setting '{setting}'. Expected one of all, text, image, none."
        )

    if distractors.empty:
        return None
    return distractors

def _safe_identifier(series: Optional[pd.Series]) -> str:
    if series is None:
        return "none"

    id = series["id"]
    return "".join(
        c if c.isalnum() or c in {"-", "_"} else "_"
        for c in str(id)
    )

################################################################################################
# PROMPTER CREATION
################################################################################################

def create_prompter(dataset_name, model, max_tokens, temperature, top_p):
    if dataset_name in DATASETS:
        if args.batch_submit:
            class_name = DATASETS[dataset_name]["batch_submit_prompter_class"]
        else:
            class_name = DATASETS[dataset_name]["prompter_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model, max_tokens, temperature, top_p)

    raise ValueError(f"Unknown Dataset '{dataset_name}'")

################################################################################################
# MODEL CREATION
################################################################################################

def create_model(model_name):
    if model_name in MODELS:
        if args.batch_submit:
            if "batch_submit_model_class" not in MODELS[model_name]:
                raise ValueError(f"Model {model_name} does not have an associated BatchSubmitModel!")
            if args.batch_submit_output_filename is None:
                raise ValueError("--batch-submit-output-filename arg is required when --batch-submit is True!")
            class_name = MODELS[model_name]["batch_submit_model_class"]
        elif args.batch_retrieve:
            if "batch_retrieve_model_class" not in MODELS[model_name]:
                raise ValueError(f"Model {model_name} does not have an associated BatchRetrieveModel!")
            if args.batch_retrieve_response_filename is None:
                raise ValueError("--batch-retrieve-response-filename arg is required when --batch-retrieve is True!")
            class_name = MODELS[model_name]["batch_retrieve_model_class"]
        else:
            class_name = MODELS[model_name]["model_class"]

        cls = getattr(sys.modules[__name__], class_name)
        model = cls(model_name)
        if args.batch_submit:
            model.set_filename(args.batch_submit_output_filename)
        if args.batch_retrieve:
            model.load_data(args.batch_retrieve_response_filename)
        return model

    raise ValueError(f"Unknown Model '{model_name}'")

################################################################################################
# SETUP
################################################################################################

dataset_config = _get_dataset_config(args.dataset)
question_formats = args.question_formats or dataset_config.get(
    "default_question_formats", []
)
if not question_formats:
    raise ValueError(f"No question formats provided for dataset '{args.dataset}'.")

# Load scenarios and distractors
scenarios = load_scenarios(args.dataset, args.eval_num_scenarios)
supports_distractors = dataset_config.get("supports_distractors", True)
if supports_distractors:
    distractors = load_distractors(args.distractors)
else:
    if args.distractors.lower() != "none":
        print(
            f"[Setup] Distractors disabled for dataset '{args.dataset}'. "
            f"Ignoring requested setting '{args.distractors}'."
        )
    distractors = None

print(
    f"[Setup] Experiment '{args.experiment_name}' | Dataset '{args.dataset}' | "
    f"Scenarios: {len(scenarios)} | Question formats: {', '.join(question_formats)}"
)
if distractors is None:
    print("[Setup] Running without distractors.")
else:
    print(f"[Setup] Loaded {len(distractors)} distractors ({args.distractors}).")

default_temperature = dataset_config.get("default_temperature", 1.0)
default_top_p = dataset_config.get("default_top_p", 1.0)
temperature = args.eval_temp if args.eval_temp is not None else default_temperature
top_p = args.eval_top_p if args.eval_top_p is not None else default_top_p
print(f"[Setup] Sampling params -> temperature={temperature}, top_p={top_p}")

# Create result folders
path_model = (
        PATH_RESULTS
        / args.experiment_name
        / f"{args.dataset}_raw"
        / args.model_name.split("/")[-1]
)
for question_format in question_formats:
    path_model_questiontype = path_model / question_format
    os.makedirs(path_model_questiontype, exist_ok=True)

################################################################################################
# RUN EVALUATION
################################################################################################

# Clean memory
gc.collect()
torch.cuda.empty_cache()

# Create model
model = create_model(args.model_name)

# Create prompter
prompter = create_prompter(
    args.dataset,
    model,
    args.eval_max_tokens,
    temperature,
    top_p,
)

# Run experiment
def run_experiment(scenario_series: pd.Series, distractor_series: Optional[pd.Series]):
    for question_format in question_formats:
        s_id = _safe_identifier(scenario_series)
        d_id = _safe_identifier(distractor_series)
        result_path = (
                path_model
                / question_format
                / f"s_{s_id}_d_{d_id}.pickle"
        )
        if os.path.exists(result_path):
            continue

        # Retry up to 5 times to handle transient API errors and model loading issues
        success = False
        tries = 0
        while not success and tries < 5:
            try:
                results = prompter.prompt(
                    question_format=question_format,
                    scenario_series=scenario_series,
                    distractor_series=distractor_series,
                )
                success = True  # Set to True to exit the loop on success
            except ValueError as e:
                tries += 1
                print(f"Caught exception: {e}")
            except Exception as e:
                tries += 1
                print(f"Caught an unexpected exception: {e}")
            finally:
                if not success:
                    results = {}

        with open(result_path, "wb") as f:
            pickle.dump(pd.DataFrame(results), f, protocol=0)

# First pass: run each scenario without any distractor (baseline condition)
for i_s, scenario_series in tqdm(
        scenarios.iterrows(),
        total=len(scenarios),
        position=0,
        ncols=100,
        leave=True,
        desc=f"Moral Distractors Eval: {model.get_model_id()}",
):
    run_experiment(scenario_series, None)

# Second pass: run every (scenario, distractor) pair
if distractors is not None:
    for (i_s, scenario_series), (i_d, distractor_series) in tqdm(
            itertools.product(scenarios.iterrows(), distractors.iterrows()),
            total=len(scenarios) * len(distractors),
            position=0,
            ncols=100,
            leave=True,
            desc=f"Moral Distractors Eval: {model.get_model_id()}",
    ):
        run_experiment(scenario_series, distractor_series)
