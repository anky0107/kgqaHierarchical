# Exp 0: Flat BERT Baseline

## Objective
Establish the bare-minimum performance of a naive transformer taking in a text string and predicting relations independently without structural bounds.

## Architecture
- **Encoder**: `bert-base-uncased` (110M parameters).
- **Head**: A single `nn.Linear(768, num_relations)`.
- **Logic**: For a sequence length of 4 hops, the model makes 4 independently evaluated argmax predictions against all 916 possible Freebase relations.

## Limitations
- **No Coherence**: Predicting Hop 2 does not fundamentally realize what was predicted at Hop 1, leading to dead-end graph paths.
- **Fixed Length**: It rigidly predicts exactly 4 hops, artificially padding 1 and 2-hop questions with noise.
- **Accuracy**: Maxed out at `31.83%` on Test.
