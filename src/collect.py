"""Aggregate per-scenario pickle result files from evaluate.py into a single CSV per model.

Walks ``data/responses/<experiment>/<dataset>_raw/``, loads ``.pickle`` files whose
names contain a modality hint (``txt``, ``img``, or ``none`` for baseline ``d_none``),
concatenates them, and writes one CSV per ``model_id`` under ``data/csv_results/``.
This matches ``evaluate.py`` filenames ``s_<scenario>_d_<distractor>.pickle`` with
current distractor ids (``txt_*``, ``img_*``)."""

import os
import pickle
import argparse
import pandas as pd

from src.config import PATH_RESULTS
from src.config import PATH_CSV_RESULTS


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Collecting Results")
parser.add_argument(
    "--experiment-name",
    default="test",
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset",
    default="moralchoice_low_ambiguity",
    type=str,
    help="Dataset key (same as evaluate.py: moralchoice_high_ambiguity, moralchoice_low_ambiguity, normbank, reddit)",
)
parser.add_argument(
    "--distractors", default="all", type=str, help="Which distractors to collect (image, text, all, none)"
)


args = parser.parse_args()


################################################################################################
# SETUP
################################################################################################
path_results = f"{PATH_CSV_RESULTS}/{args.experiment_name}/{args.dataset}/{args.distractors}"
path_results_raw = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}" + "_raw"

################################################################################################
# RESPONSE COLLECTION
################################################################################################
# Collect all pickle result files
if args.distractors == "image":
    distractors = "img"
elif args.distractors == "text":
    distractors = "txt"
elif args.distractors == "none":
    distractors = "none"
else:
    distractors = "all"

results = []
for path, subdirs, files in os.walk(path_results_raw):
    for name in files:
        if name[-7:] == ".pickle" and (distractors in name or distractors == "all"):
            path_file = os.path.join(path, name)

            with open(path_file, "rb") as f:
                tmp = pickle.load(f)
                results.append(tmp)

if not results:
    raise SystemExit(
        f"No matching .pickle files under {path_results_raw!r}. "
        "Check --experiment-name, --dataset, and --distractors."
    )

df_results = pd.concat(results)

# Store one csv per model
if not os.path.exists(path_results):
    os.makedirs(path_results)

for model_id in df_results["model_id"].unique():
    results_model = df_results.loc[df_results["model_id"] == model_id]
    results_model.to_csv(
        f"{path_results}/{model_id.split('/')[0]}_{model_id.split('/')[-1]}.csv"
    )
