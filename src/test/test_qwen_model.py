from src.models.qwen_model import QwenModel
from src.prompters.prompt import Distractor, Modality

qwen_model: QwenModel = QwenModel("Qwen/Qwen2-VL-2B-Instruct")
distractor: Distractor = {
    "id": "test",
    "modality": Modality.IMAGE,
    "file_path": "data/img_distractor_data/negative/Car crash 1.jpg"
}
response = qwen_model.query(
    system_prompt="",
    user_prompt="What's in this image? In this image is ",
    max_tokens=256,
    temperature=0.5,
    top_p=0.5,
    distractor=distractor
)

print(response.answer_raw)