import pandas as pd
from src.models.gemma_model import GemmaModel
from src.prompters.moralchoice_prompter import MoralChoicePrompter

distractor_series = pd.Series({
    "distractor_id": "test",
    "modality": "text",
    "text": "test",
    "img_path": "test"
})

scenario_series = pd.Series({
    "scenario_id": "test",
    "context": "Say hi",
    "action1": "test",
    "action2": "test"
})

gemma_model: GemmaModel = GemmaModel("google/gemma-3-4b-it")
prompter = MoralChoicePrompter(
    model=gemma_model,
    max_tokens=10,
    temperature=0.1,
    top_p=0.5
)

result = prompter.prompt(scenario_series, "ab_moralchoice", distractor_series)
print(result)