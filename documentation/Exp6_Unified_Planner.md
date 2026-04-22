# Exp 6: Unified Adaptive CHCP

## Objective
Bring back the Domain classification from Exp 3, but merge it softly into the CHCP pipeline to anchor semantic confidence without hard-masking breaking the chain.

## Architecture
- **Confidence Head**: The [CLS] token outputs a scalar $p_{conf}$, projecting how inherently difficult the model believes the question is.
- **Unified Objective**: The total multi-task loss is dynamically synthesized: $\mathcal{L}_{total} = \mathcal{L}_{relation} + \mathcal{L}_{stop} + \mathcal{L}_{domain}$. 
- **Result**: Accuracy dipped slightly to `51.70%` due to multi-task objective contention in the smaller BERT backbone conflicting over the latent vector space.
