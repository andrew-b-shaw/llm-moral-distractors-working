from openai import OpenAI

from src.models.model_utils import get_api_key

api_key = get_api_key("openai")
client = OpenAI(api_key=api_key)

response = client.batches.create(
    input_file_id="/storage/llm-moral-distractors/gpt-4_moralchoice_low_ambiguity.jsonl",
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
        "description": "gpt-4.1-moralchoice_low_ambiguity"
    }
)

print(response)
