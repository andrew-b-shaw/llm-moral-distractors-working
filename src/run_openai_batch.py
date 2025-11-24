import json

from openai import OpenAI
from src.models.model_utils import get_api_key

api_key = get_api_key("openai")
client = OpenAI(api_key=api_key)
input_filename = "gpt-4.1-reddit-batch"
input_filepath = f"/storage/llm-moral-distractors/{input_filename}.jsonl"

# Verify batch input file
# with open(input_filepath, 'r') as f:
#     print(f.readline())
#     print(f.readline())
# with open(input_filepath, 'r') as f:
#     num_lines = 0
#     for line in f:
#         num_lines += 1
#     print(num_lines)

# Upload batch input file
# batch_input_file = client.files.create(
#     file=open(input_filepath, "rb"),
#     purpose="batch"
# )
# print(batch_input_file)

# Create batch
# input_file_id = "file-1had4dyjYdmsnAxdnQTCgm"
# response = client.batches.create(
#     input_file_id=input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={
#         "description": input_filename
#     }
# )
# print(response)

# Check batch status
batch_id = "batch_6924b39476c08190970306b73586c965"
batch = client.batches.retrieve(batch_id)
print(batch)

# Check errors
# error_file_id = "file-WMSzBa3MTiT1yF3ABTQ5gH"
# file_response = client.files.content(error_file_id)
# print(file_response.text)

# Retrieve results
# output_file_id = "file-VmzUiZhkLaLzhYS8UrBzf1"
# file_response = client.files.content(output_file_id)
#
# output_filename = f"{input_filename}-output.jsonl"
# with open(output_filename, "wb") as f:
#     f.write(file_response.content)
#
# output = file_response.text.split("\n")
# print(json.dumps(json.loads(output[0]), indent=4))
