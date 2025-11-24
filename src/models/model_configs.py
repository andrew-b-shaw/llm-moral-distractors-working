API_TIMEOUTS = [1, 2, 4, 8, 16, 32]

####################################################################################
# MODELS DICT
####################################################################################
MODELS = {
    "openai/gpt-4.1": {
        "company": "openai",
        "model_class": "OpenAIModel",
        "model_name": "openai/gpt-4.1",
        "8bit": None,
        "likelihood_access": False,
        "endpoint": "ChatCompletion",
    },
    "openai/gpt-4.1-batch-submit": {
        "company": "openai",
        "model_class": "OpenAIBatchSubmitModel",
        "model_name": "openai/gpt-4.1-batch-submit",
        "8bit": None,
        "likelihood_access": False,
        "endpoint": "ChatCompletion",
        "output_filepath": "gpt-4.1-reddit-batch.jsonl"  # set manually
    },
    "openai/gpt-4.1-batch-retrieve": {
        "company": "openai",
        "model_class": "OpenAIBatchRetrieveModel",
        "model_name": "openai/gpt-4.1-batch-retrieve",
        "8bit": None,
        "likelihood_access": False,
        "endpoint": "ChatCompletion",
        "input_filepath": "gpt-4.1-reddit-batch-output.jsonl"  # set manually
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
    "ollama/gemma3-4b": {
        "company": "ollama",
        "model_class": "OllamaModel",
        "model_name": "ollama/gemma3-4b",
        "ollama_model": "gemma3:4b",
        "8bit": False,
        "likelihood_access": False,
        "endpoint": "ollama",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "company": "Meta",
        "model_class": "LlamaModel",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
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
