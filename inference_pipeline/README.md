# Experiment 7 Inference Pipeline

This folder contains a minimal working example of the inference pipeline for **Experiment 7 (RoBERTa-Large Unified Planner)**.

## Structure
- `model.py`: Self-contained model definition for `ScaledUnifiedPlanner`.
- `exp7_inference.py`: Main inference script that handles:
    - Loading RoBERTa-Large and best-performing weights (`exp7_roberta_best.pt`).
    - Knowledge Graph loading and traversal.
    - Step-by-step transparent execution.

## How to Run
```bash
python inference_pipeline/exp7_inference.py
```

## Inference Stages
The pipeline executes in 5 distinct stages:
1. **Question Encoding**: Uses RoBERTa-Large to generate a high-dimensional question embedding.
2. **Progressive Constraints**: Predicts the question domain (e.g., `people`, `sports`) and a confidence score.
3. **Cross-Hop Reasoning**: Uses a Transformer Encoder to refine representations for each hop.
4. **Relation Prediction**: Predicts the most likely KG relation for each step.
5. **KG Execution**: Physically traverses the knowledge graph from the topic entity to find the answer.

## Key Files Reused
- `checkpoints/exp7_roberta_best.pt`: Model weights.
- `data/processed_kg/augmented_kg.pt`: Pre-processed KG subgraph.
- `data/master_mid2name.json`: Entity name mapping.
- `data/processed_entity/relation2id.pt`: Vocabulary mappings.
