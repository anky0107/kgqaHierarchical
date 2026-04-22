# Exp 4: Cross-Hop Coherence Planning (CHCP)

## Objective
Fix the cascading failure of Exp 3 by doing "Soft" hierarchical refinement instead of "Hard" masking.

## Architecture
- **Hop Embeddings**: Introduces learnable positional vectors: $\mathbb{E}_{hop} \in \mathbb{R}^{4 \times D}$. These give the sequence spatial awareness (knowing you are at hop 2 vs hop 4).
- **Transformer Refinement Layer**: The sequence of positional vectors is passed through an internal 3-layer `nn.TransformerEncoder`. This allows Hop 3 to 'look' at Hop 1 through self-attention before predicting its relation.
- **Adaptive Stopping**: Introduces an independent `stop_logits` head driven by Binary Cross Entropy, allowing the model to dynamically output $p(stop) > 0.5$ when it mathematically decides the sequence is complete.

## Results
- **Accuracy**: Skyrocketed to `55.50%` on Test. The introduction of cohesive sequence self-attention fundamentally solved early-hop drifting.
