####################################################################################
# MODEL CREATOR
####################################################################################
import sys

from src.models.models import MODELS
from src.models.openai_model import OpenAIModel
from src.models.gemma_model import GemmaModel
from src.models.llama_model import LlamaModel
from src.models.ollama_model import OllamaModel
from src.models.qwen_model import QwenModel
from src.models.qwen_vl_model import QwenVLModel

OpenAIModel = OpenAIModel
GemmaModel = GemmaModel
LlamaModel = LlamaModel
OllamaModel = OllamaModel
QwenModel = QwenModel
QwenVLModel = QwenVLModel


def create_model(model_name):
    """Init Models from model_name only"""
    if model_name in MODELS:
        class_name = MODELS[model_name]["model_class"]
        cls = getattr(sys.modules[__name__], class_name)
        return cls(model_name)

    raise ValueError(f"Unknown Model '{model_name}'")
