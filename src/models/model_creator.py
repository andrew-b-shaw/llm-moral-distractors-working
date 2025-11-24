####################################################################################
# MODEL CREATOR
####################################################################################
import sys

from src.models.model_configs import MODELS
from src.models.openai_model import OpenAIModel, OpenAIBatchSubmitModel, OpenAIBatchRetrieveModel
from src.models.gemma_model import GemmaModel
from src.models.llama_model import LlamaModel
from src.models.ollama_model import OllamaModel
from src.models.qwen_model import QwenModel
from src.models.qwen_vl_model import QwenVLModel


def create_model(model_name):
    """Init Models from model_name only"""
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model_name)

    raise ValueError(f"Unknown Model '{model_name}'")
