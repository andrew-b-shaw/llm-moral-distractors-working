# llm_moral_plasticity

## Research Question:
Are the moral judgements of LLMs robust to morally irrelevant situational distractors? We investigate this question as part of our paper for CSE 599 J1, "Moral Distrators for LLMs." This repository contains the code and data used to run our experiments.


## Useful Commands

Run evaluation with specified experiment, dataset, model, and question types:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
  --experiment-name "moraltest" \
  --dataset "moralchoice_high_ambiguity" \
  --model "google/flan-t5-small" \
  --question-types "ab" \
  --eval-nb-samples 5 \
  --eval-max-tokens 1
```

Run result collection on experiment and specified dataset:

```bash
python -m src.collect \
  --experiment-name "moraltest" \
  --dataset "moralchoice_high_ambiguity"
```

