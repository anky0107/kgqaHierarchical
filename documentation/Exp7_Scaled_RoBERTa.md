# Exp 7: Scaled Unified Planner (RoBERTa-Large)

## Objective
Un-bottleneck the parameter ceiling from Exp 6 by scaling the underlying encoder architecture and hidden dimension width.

## Architecture
- **Encoder**: Swapped to `roberta-large` (355M Parameters).
- **Hidden Dim Strategy**: We maintain a projected $D=512$ space for the refinement transformers to remain memory efficient on RTX consumer hardware.
- **Scaling Depth**: The internal refinement layer is pushed to 4 Transformer heads.

## Results
- **Accuracy**: Jumped to `57.28%`. Converged smoothly without domain collision due to larger representation volume.
