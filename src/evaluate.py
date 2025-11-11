import itertools
import os
import pickle
import json
import argparse
import sys

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.models.model_creator import create_model
from src.prompters.moralchoice_prompter import MoralChoicePrompter
from src.prompters.reddit_prompter import RedditPrompter

from src.config import PATH_RESULTS, PATH_RESPONSE_TEMPLATES


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="LLM Evaluation on MoralChoice")
parser.add_argument(
    "--experiment-name",
    default="test",
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset", default="high", type=str, help="Dataset to evaluate (moralchoice_high_ambiguity, moralchoice_law_ambiguity, reddit)"
)
parser.add_argument(
    "--distractors", default=False, type=bool, help="Whether to use distractors"
)
parser.add_argument(
    "--model-name",
    default="openai/text-babbage-001",
    type=str,
    help="Model to evalute --- see models.py for an overview of supported models",
)
parser.add_argument(
    "--question-formats",
    default=["ab"],
    type=str,
    help="Question Templates to evaluate",
    nargs="+",
)
parser.add_argument(
    "--eval-top-p", default=1.0, type=float, help="Top-P parameter for top-p sampling"
)
parser.add_argument(
    "--eval-temp", default=1.0, type=float, help="Temperature for sampling"
)
parser.add_argument(
    "--eval-max-tokens",
    default=200,
    type=int,
    help="Max. number of tokens per completion",
)
parser.add_argument(
    "--eval-num-samples", default=1, type=int, help="Num. of samples per question form"
)
parser.add_argument(
    "--eval-num-scenarios", default=-1, type=int, help="Num. of scenarios to evaluate"
)

args = parser.parse_args()


################################################################################################
# PROMPTER CREATOR
################################################################################################
DATASETS= {
    "moralchoice_high_ambiguity": {
        "prompter_class": "MoralChoicePrompter"
    },
    "moralchoice_low_ambiguity": {
        "prompter_class": "MoralChoicePrompter"
    },
    "reddit": {
        "prompter_class": "RedditPrompter"
    }
}

MoralChoicePrompter = MoralChoicePrompter
RedditPrompter = RedditPrompter

def create_prompter(dataset_name, model, max_tokens, temperature, top_p):
    """Init Models from model_name only"""
    if dataset_name in DATASETS:
        class_name = DATASETS[dataset_name]["prompter_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model, max_tokens, temperature, top_p)

    raise ValueError(f"Unknown Dataset '{dataset_name}'")


################################################################################################
# SETUP
################################################################################################

# Load scenarios
scenarios = pd.read_csv(f"data/scenarios/{args.dataset}.csv")
scenarios = scenarios[:args.eval_num_scenarios] if args.eval_num_scenarios > 0 else scenarios
# TODO: add support for reddit dataset

if args.distractors:
    distractors = pd.read_csv(f"data/distractors.csv")
else:
    distractors = None

# Creates result folders
path_model = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}_raw/{args.model_name.split('/')[-1]}"
for question_format in args.question_formats:
    path_model_questiontype = path_model + f"/{question_format}"
    if not os.path.exists(path_model_questiontype):
        os.makedirs(path_model_questiontype)


################################################################################################
# RUN EVALUATION
################################################################################################
model = create_model(args.model_name)
prompter = create_prompter(args.dataset, model, args.eval_max_tokens, args.eval_temp, args.eval_top_p)

if distractors is None:
    for i_s, scenario in tqdm(
        scenarios.iterrows(),
        total=len(scenarios),
        position=0,
        ncols=100,
        leave=True,
        desc=f"No Moral Distractors Eval: {model.get_model_id()}",
    ):
        for question_format in args.question_formats:
            # No distractor condition
            results = prompter.prompt(
                question_format=question_format,
                scenario_series=scenario,
                distractor_series =None
            )

            with open(
                    f'{path_model}/{question_format}/scenario_{scenario["id"]}_no_distractor.pickle',
                    "wb",
            ) as f:
                pickle.dump(pd.DataFrame(results), f, protocol=0)
else:
    for (i_s, scenario), (i_d, distractor) in tqdm(
        itertools.product(scenarios.iterrows(), distractors.iterrows()),
        total=len(scenarios) * len(distractors),
        position=0,
        ncols=100,
        leave=True,
        desc=f"Moral Distractors Eval: {model.get_model_id()}"
    ):
        for question_format in args.question_formats:
            results = prompter.prompt(
                question_format=question_format,
                scenario_series=scenario,
                distractor_series = distractor
            )

            with open(
                    f'{path_model}/{question_format}/scenario_{scenario["id"]}_distractor_{distractor["id"]}.pickle',
                    "wb",
            ) as f:
                pickle.dump(pd.DataFrame(results), f, protocol=0)
