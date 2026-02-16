from __future__ import annotations
from src.models.models import LanguageModelResponse
from src.prompters.normbank_prompter import NormBankPrompter
from src.prompters.prompt import Prompt


class NormBankBatchSubmitPrompter(NormBankPrompter):
    def post_process(
        self,
        prompt: Prompt,
        response: LanguageModelResponse
    ) -> dict[str, any]:
        return {}
