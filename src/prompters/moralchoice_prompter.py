import pandas as pd

from data.templates.question_templates import QUESTION_TEMPLATES
from data.templates.response_templates import A_TOKENS, B_TOKENS, YES_TOKENS, NO_TOKENS
from src.models.models import LanguageModelResponse
from src.prompters.prompter import Scenario, Prompt, Prompter
from src.prompters.distractor import Distractor


class MoralChoiceScenario(Scenario):
    actions: list[str]


class MoralChoicePrompt(Prompt):
    question_type: str
    question_ordering: int


class MoralChoicePrompter(Prompter[MoralChoicePrompt]):
    def generate_prompt(
        self,
        scenario: MoralChoiceScenario,
        question_type: str,
        question_ordering: int,
        distractor: Distractor | None = None
    ) -> MoralChoicePrompt:
        question_template = QUESTION_TEMPLATES[question_type]
        prompt: MoralChoicePrompt = {
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": question_template["system"],
            "user_prompt": question_template["user"].format(
                scenario["context"],
                scenario["actions"][question_ordering],
                scenario["actions"][1 - question_ordering]
            ),
            "question_type": question_type,
            "question_ordering": question_ordering
        }
        return prompt

    def pre_process(
        self,
        scenario_series: pd.Series,
        question_type: str,
        distractor_series: pd.Series | None,
        distractor_modality: str | None
    ) -> list[MoralChoicePrompt]:
        # Create Distractor
        distractor: Distractor | None = {
            "modality": distractor_modality,
            "distractor_id": distractor_series["distractor_id"],
            "text": distractor_series["text"] if distractor_modality == "text" else None,
            "img": distractor_series["img_path"] if distractor_modality == "img" else None
        } if distractor_series else None

        # Create Scenario
        scenario: MoralChoiceScenario = {
            "scenario_id": scenario_series["scenario_id"],
            "context": scenario_series["context"],
            "actions": [scenario_series["action1"], scenario_series["action2"]]
        }

        # Generate prompts with all orderings
        prompts = []
        for question_ordering in [0, 1]:
            prompt = self.generate_prompt(
                scenario=scenario,
                distractor=distractor,
                question_type=question_type,
                question_ordering=question_ordering
            )
            prompts.append(prompt)
        return prompts

    def post_process(
        self,
        prompt: MoralChoicePrompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        if prompt["question_type"] == "ab":
            # A/B question format
            action_tokens_dict = {
                "A": A_TOKENS,
                "B": B_TOKENS
            }
            action_mapping_dict = {
                "A": prompt["question_ordering"],
                "B": 1 - prompt["question_ordering"]
            }
        else:
            # YES/NO question format
            action_tokens_dict = {
                "YES": YES_TOKENS,
                "NO": NO_TOKENS
            }
            action_mapping_dict = {
                "YES": prompt["question_ordering"],
                "NO": 1 - prompt["question_ordering"]
            }
        # TODO: incorporate LLM-as-judge for open-ended generation

        result = {
            "model_id": self.model.get_model_id(),
            "question_type": prompt["question_type"],
            "question_ordering": prompt["question_ordering"],
            "system_prompt": prompt["system_prompt"],
            "user_prompt": prompt["user_prompt"],
            "eval_technique": "top_p_sampling",
            "eval_top_p": self.eval_top_p,
            "eval_temperature": self.eval_temp,
            "image_path": prompt["distractor"]["image_path"],
            "answer_raw": response.answer_raw,
            "answer": response.answer,
            "action_0_prob": 0.0,
            "action_1_prob": 0.0
        }

        for action, tokens in action_tokens_dict.items():
            for token in tokens:
                result[f"action_{action_mapping_dict[action]}_prob"] += response.get_answer_prob(token)

        return result