from src.models.gemma_model import GemmaModel

gemma_model: GemmaModel = GemmaModel("google/gemma-3-4b-pt")
response = gemma_model._generate_with_image(
    image_path="/storage/llm-moral-distractors/data/img_distractor_data/negative/Car crash 1.jpg",
    system_prompt="",
    user_prompt="In this image is ",
    max_new_tokens=256,
    temperature=0.5,
    top_p=0.5
)

print(response.answer_raw)