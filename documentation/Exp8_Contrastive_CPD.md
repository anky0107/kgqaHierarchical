# Exp 8: Contrastive Path Discrimination (CPD)

## Objective
The Exp 7 model naturally confuses semantically identical boundaries (e.g., confusing `film.director` with `film.writer`). We must mechanically force a margin between these overlapping spaces.

## Architecture
- **Dynamic Hard Negative Mining**: We scan the model's logits at Hop $h$ and extract the highest-scoring *wrong* relations in real-time. We swap the wrong relation into the sequence to create an Adversarial Path ($p^-_i$).
- **Path-Level InfoNCE**: We construct a Contrastive Loss equation:
  $$ \mathcal{L}_{cpd} = -\log \frac{\exp(s(q, p^+) / \tau)}{\sum_{i=0}^4 \exp(s(q, p^-_i) / \tau)} $$
- **Implementation**: The loss optimizes over the sum of internal probabilities, forcing the margin between the gold path and the highest-potential trap path to expand exponentially.

## Results
- **Accuracy**: While Hits@1 remained stable at `56.76%`, Hits@3 massively surged to **85.24%**. Contrastive learning successfully trapped the truthful answer systematically within the Top-3 logical choices.
