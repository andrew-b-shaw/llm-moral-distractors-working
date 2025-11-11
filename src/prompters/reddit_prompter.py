import pandas as pd

from data.templates.question_templates import QuestionTemplate, QUESTION_TEMPLATES
from src.prompters.prompter import Prompter
from src.prompters.prompt import Prompt, Scenario, Distractor


class RedditPrompter(Prompter[Prompt]):
    def generate_prompt(
            self,
            scenario: Scenario,
            question_type: str,
            distractor: Distractor | None,
    ) -> Prompt:
        question_template: QuestionTemplate = QUESTION_TEMPLATES[question_type]
        prompt: Prompt = {
            "scenario": scenario,
            "distractor": distractor,
            "system_prompt": question_template["system"],
            "user_prompt": question_template["user"].format(scenario["context"])
        }

        return prompt

    def pre_process(
            self,
            scenario_series: pd.Series,
            question_type: str,
            distractor_series: pd.Series | None,
            distractor_modality: str | None
    ) -> list[Prompt]:
        # Create Distractor
        distractor: Distractor | None = {
            "modality": distractor_modality,
            "distractor_id": distractor_series["distractor_id"],
            "text": distractor_series["text"] if distractor_modality == "text" else None,
            "img": distractor_series["img_path"] if distractor_modality == "img" else None
        } if distractor_series else None

        # Create Scenario
        scenario: Scenario = {
            "scenario_id": scenario_series["scenario_id"],
            "context": scenario_series["selftext"]
        }

        # Generate prompt
        prompt = self.generate_prompt(
            scenario=scenario,
            question_type=question_type,
            distractor=distractor
        )
        return [prompt]

    def post_process(
            self,
            prompt: Prompt,
            response: dict[str, any]
    ) -> dict[str, any]:
        # TODO: complete reddit post-processing
        pass