import pandas as pd

from data.templates.question_templates import QUESTION_TEMPLATES
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
            response: dict[str, any]
    ) -> dict[str, any]:
        if prompt["question_type"] == "ab":
            action_mapping_dict = {
                "A": prompt["question_ordering"],
                "B": 1 - prompt["question_ordering"]
            }
        else:
            action_mapping_dict = {
                "YES": prompt["question_ordering"],
                "NO": 1 - prompt["question_ordering"]
            }
        action_mapping = lambda x: action_mapping_dict[x]  # TODO: incorporate LLM-as-judge for open-ended generation

        # TODO: complete MoralChoice post-processing
        pass