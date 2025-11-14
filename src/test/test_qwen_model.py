from src.models.qwen_model import QwenModel
from src.prompters.prompt import Distractor, Modality

qwen_model: QwenModel = QwenModel("Qwen/Qwen3-1.7B")
# distractor: Distractor = {
#     "id": "test",
#     "modality": Modality.IMAGE,
#     "file_path": "C:\\Users\\andre\\IdeaProjects\\llm-moral-distractors\\data\\img_distractor_data\\negative\\Car crash 1.jpg"
# }
response = qwen_model.query(
    system_prompt="",
    user_prompt="What is the capital of France? ",
    max_tokens=256,
    temperature=0.5,
    top_p=0.5,
    distractor=None
)

print(len(response._output.logits))