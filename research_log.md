# KGQA Research Experiment Log

This document tracks all experimental runs, configurations, and results.

## Phase 1: Baseline & Architectural Innovations (March 2026)

### Experiment 0: Flat Relation Baseline
- **Configuration**: BERT-base with a 916-class relation classification head.
- **Results**:
  - Hits@1: 0.3099
  - Hits@3: 0.4355
  - Hop Accuracy: 0.3307
- **Notes**: Performance is solid on 1-hop but degrades rapidly on multi-hop as relations are predicted independently.

### Experiment 1: Domain-Restricted Search
- **Configuration**: BERT-based domain classifier + filtered beam search.
- **Results**: Hits@1 (component) 0.7083.
- **Notes**: Good for 1-hop, but domain classification acts as a bottleneck for long-tail entities.

### Experiment 2: Contrastive Path Discrimination (CPD)
- **Configuration**: InfoNCE Loss with relation-semantic hard negative mining.
- **Results**: 97% training accuracy on path discrimination.
- **Notes**: Proves that the model can successfully distinguish between semantically similar relations ("born_at" vs "died_at").

### Experiment 3: Progressive Constraint Tightening (PCT)
- **Configuration**: Multi-head model with scalar confidence and coarse-to-fine planning.
- **Results**: 
  - Hits@1: 0.3402
  - Hits@3: 0.4578
  - 1-hop H@1: 0.5857
- **Notes**: Outperforms Exp 0 by adding per-hop confidence, but lacks cross-step coherence.

### Experiment 4: Cross-Hop Coherence Planning (CHCP)
- **Configuration**: Bidirectional Transformer Encoder over 4 initial hop slots.
- **Results**:
  - **Overall Hits@1: 0.5557 (SOTA-Comparable)**
  - Hits@3: 0.7579
  - 2-hop H@1: 0.5614
  - 3-hop H@1: 0.5587
- **Notes**: Currently our strongest model. Matches DRKG (0.561) and ChatKBQA (0.555).

---

## Phase 2: SOTA Push (Current)

### Experiment 4-RL: CHCP + PPO Fine-tuning
- **Objective**: Improve answer-hit rate via Reinforcement Learning on the KG subgraph.
- **Status**: [RUNNING]
- **Epoch 0 Results**: Avg Reward -0.238, Success Rate 0.005. (Exploration phase).

### Experiment 6: Unified Adaptive Planner
- **Objective**: Merge Confidence (Exp 3) and Coherence (Exp 4) into a single unified architecture.
- **Status**: [TRAINING]
- **Intermediate Results**:
  - Epoch 0 Dev Rel Loss: 1.8456
- **Code**: `train/exp6_unified.py`

### Experiment 7: Scaling to RoBERTa-Large
- **Objective**: Push the architecture to its capacity limit.
- **Status**: [PLANNED]
