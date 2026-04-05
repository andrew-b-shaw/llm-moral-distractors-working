"""Registry of supported model configurations.

Each entry maps a model identifier to provider metadata, Python class names,
backend endpoint settings, and capability flags.
API_TIMEOUTS defines the exponential backoff schedule for API retries."""

API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

MODELS = {
    "openai/gpt-4.1": {
        "company": "openai",
        "model_class": "OpenAIModel",
        "model_name": "openai/gpt-4.1",
        "8bit": None,
        "likelihood_access": False,
        "endpoint": "ChatCompletion",
        "batch_submit_model_class": "OpenAIBatchSubmitModel",
        "batch_retrieve_model_class": "OpenAIBatchRetrieveModel"
    },
    "Qwen/Qwen3-VL-2B-Instruct": {
        "company": "AliBaba",
        "model_class": "QwenVLModel",
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "Qwen/Qwen3-4B": {
        "company": "AliBaba",
        "model_class": "QwenModel",
        "model_name": "Qwen/Qwen3-4B",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "Qwen/Qwen3-4B-Base": {
        "company": "AliBaba",
        "model_class": "QwenModel",
        "model_name": "Qwen/Qwen3-4B-Base",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "Qwen/Qwen3-1.7B": {
        "company": "AliBaba",
        "model_class": "QwenModel",
        "model_name": "Qwen/Qwen3-1.7B",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "Qwen/Qwen3-0.6B": {
        "company": "AliBaba",
        "model_class": "QwenModel",
        "model_name": "Qwen/Qwen3-0.6B",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "google/gemma-3-270m-pt": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-270m-pt",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-270m-it": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-270m-it",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-1b-pt": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-1b-pt",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-1b-it": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-1b-it",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-4b-pt": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-4b-pt",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-4b-it": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-4b-it",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-12b-pt": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-12b-pt",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-12b-it": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-12b-it",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-27b-pt": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-27b-pt",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "google/gemma-3-27b-it": {
        "company": "google",
        "model_class": "GemmaModel",
        "model_name": "google/gemma-3-27b-it",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "company": "Meta",
        "model_class": "LlamaModel",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "meta-llama/Llama-3.2-3B": {
        "company": "Meta",
        "model_class": "LlamaModel",
        "model_name": "meta-llama/Llama-3.2-3B",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "company":"Meta",
        "model_class": "LlamaModel",
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "8bit": False,
        "likelihood_access": True,
        "endpoint": None
    }
}
