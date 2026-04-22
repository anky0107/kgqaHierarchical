# Exp 3: Progressive Constraint Tightening (PCT)

## Objective
Attempt to solve the flat baseline's "dead-end" problem using hard masks. 

## Architecture
- **Dual Head Pipeline**: A `Domain Head` first classifies the high-level semantic context (e.g., `film`, `music`, `sports`). 
- **Relation Masking**: The `Relation Head` only allows logits to pass if the relation falls entirely within that predicted Domain. 
- **Graph Hard Masking**: To proceed from Hop $h$ to Hop $h+1$, the mask strictly checks if a graph edge mathematically exists between the two predicted relations.

## Limitations
- **Cascading Failure**: If the Domain Head makes a slight mistake at the very beginning of the pipeline, the Relation Head is forced to pick from a 100% wrong subset of relations, guaranteeing a 0% accuracy trace. 
- **Accuracy**: Decreased baseline to `26.85%` due to aggressive over-masking preventing path recovery.
