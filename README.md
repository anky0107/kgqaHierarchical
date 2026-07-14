# Knowledge Graph Question Answering Using Reinforcement Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**An adaptive neuro-symbolic framework for multi-hop KGQA applying Reinforcement Learning at two levels: a policy-gradient traversal controller and a DPO-aligned generative judge.**

*NIT Calicut — Ankit Rana, Dr. Chandramani Chaudhary*

</div>

---

## Overview

Multi-hop Knowledge Graph Question Answering (KGQA) requires compositional reasoning over large-scale knowledge graphs. Existing systems use static traversal strategies — fixed beam widths, exhaustive expansion — causing excessive graph exploration and error propagation. Direct RL over raw relation IDs suffers from intractably large action spaces and unstable training.

This project proposes **RLMC** (Reinforcement Learning Meta-Constraint), a three-stage adaptive neuro-symbolic framework that:

- **Reduces the RL action space** from >10,000 relation IDs to **4 interpretable width-control actions** (Tight / Medium / Loose / Stop)
- **Cuts average graph edges expanded** from ~1,912 (static) to **~14 per question** (STRL variant)
- **Achieves 42.12% Hit@1** on ComplexWebQuestions using only RoBERTa-Large (355M), without frontier-scale LLMs

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Stage I: Semantic Reasoning Planner                         │
│  RoBERTa-Large → Linear Proj (z_q) → Cross-Hop Transformer  │
│  Heads: Relation | Domain | Confidence (η_q) | Stop         │
├──────────────────────────────────────────────────────────────┤
│  Stage II: RL-Based Traversal Controller (RLMC / STRL)      │
│  State s_t = [h_h ; η_q] → Actor-Critic MLP (d_mlp=256)    │
│  Actions: Tight(k=1) | Medium(k=5) | Loose(k=50) | Stop     │
│  Training: A2C (RLMC) / PPO + InfoNCE curriculum (STRL)     │
├──────────────────────────────────────────────────────────────┤
│  Stage III: CDS — Cascading Dust Separator                   │
│  F1: MiniLM-L6 bi-encoder → top-200                         │
│  F2: MPNet path-aware ranker → top-50                        │
│  F3: Flan-T5-Base DPO generative judge → final answer       │
└──────────────────────────────────────────────────────────────┘
```

The full architecture diagram is in [`paper_kgqarl/paper/tikz/fig_detailed_dataflow.tex`](paper_kgqarl/paper/tikz/fig_detailed_dataflow.tex).

---

## Results

All results on **ComplexWebQuestions (CWQ) dev set** (test labels not publicly available; dev is the standard reported split in the KGQA literature).

| Model | Year | Reasoning Recall | Hit@1 |
|---|---|---|---|
| GraftNet | 2018 | — | 32.8% |
| EmbedKGQA† | 2020 | — | ~40.9% |
| NSM | 2021 | — | 47.6% |
| TransferNet | 2021 | — | 48.6% |
| **Planner only (no RL)** | — | ~58.0% | 37.2% |
| **STRL + CDS-DPO** | — | **75.6%** | 40.4% |
| **RLMC + CDS-DPO (Ours)** | — | 63.1% | **42.1%** |

†EmbedKGQA CWQ number approximate (primarily reported on WebQSP).

### Efficiency Metrics

| Metric | RLMC+DPO | STRL+DPO |
|---|---|---|
| Avg. Edges Expanded | ~1,912 | **~14** |
| Inference Latency | 145 ms/q | **22 ms/q** |
| Peak VRAM | 6.2 GB | **4.8 GB** |
| Path Accuracy | 41.6% | **89.4%** |
| Depth Accuracy | 65.2% | **92.2%** |

> **Key insight:** STRL achieves an order-of-magnitude reduction in traversal cost via InfoNCE semantic grounding, while RLMC achieves higher Hit@1 precision. There is a precision-recall trade-off between the two variants.

---

## Repository Structure

```
kgqaHierarchical/
├── train/
│   ├── exp6_unified.py          # Stage I: Semantic Planner training
│   ├── exp9_rlmc.py             # Stage II: RLMC A2C controller
│   ├── exp15_strl.py            # Stage II: STRL PPO + InfoNCE variant
│   ├── exp16_cds.py             # Stage III: CDS pipeline
│   ├── exp37_build_dpo_dataset.py   # Build DPO preference pairs
│   ├── exp38_train_t5_dpo.py    # F3: Flan-T5 DPO judge training
│   ├── exp26_t5_generative_s3.py    # F3: SFT reference model
│   ├── exp27_train_s2_mpnet.py  # F2: MPNet path ranker
│   └── precompute_entity_embs.py    # F1: Pre-compute MiniLM embeddings
├── eval/
│   ├── e2e_evaluate.py          # End-to-end pipeline evaluation
│   ├── extended_metrics_eval.py # Full metrics (Hit@1, Recall, Latency)
│   └── correct_hit1_eval.py     # Hit@1 evaluation
├── shared/
│   └── kg_loader.py             # Freebase LMDB graph loader
├── scripts/
│   ├── filter_freebase.py       # Build Freebase subgraph from RDF
│   ├── gen_selector_data.py     # Generate CDS training data
│   └── verify_paper_numbers.py  # Reproduce paper metrics
├── cds_pipeline/                # CDS inference pipeline modules
├── inference_pipeline/          # Full end-to-end inference
├── paper_kgqarl/                # LaTeX source for the paper
│   ├── main.tex
│   └── paper/tikz/              # All TikZ architecture diagrams
├── run_full_pipeline.py         # Full pipeline runner
├── requirements.txt
└── .gitignore
```

---

## Setup

### Requirements

- Python 3.9+
- NVIDIA GPU (≥8GB VRAM recommended; 6.2 GB minimum for RLMC)
- Freebase RDF dump (for KG construction)
- ComplexWebQuestions dataset

### Installation

```bash
git clone https://github.com/anky0107/kgqaHierarchical.git
cd kgqaHierarchical
pip install -r requirements.txt
```

### Data Preparation

1. **Build Freebase subgraph** (LMDB indexed):
```bash
python scripts/filter_freebase.py --rdf freebase-rdf-latest.gz --out data/freebase_lmdb/
```

2. **Download ComplexWebQuestions**:
```bash
# Download from https://www.tau-nlp.sites.tau.ac.il/compwebq
# Place train/dev JSON files in data/cwq/
```

---

## Training

Training follows a three-stage progressive pipeline. Each stage builds on the frozen outputs of the previous.

### Stage I — Semantic Reasoning Planner

```bash
python train/exp6_unified.py \
    --data_dir data/cwq/ \
    --kg_path data/freebase_lmdb/ \
    --output_dir checkpoints/stage1/ \
    --epochs 30 \
    --lr 1e-5 \
    --batch_size 4 \
    --grad_accum 4
```

*Trains RoBERTa-Large with Relation, Domain, Confidence, and Stop heads via multi-task supervision.*

### Stage II — RL Traversal Controller (RLMC)

```bash
python train/exp9_rlmc.py \
    --planner_ckpt checkpoints/stage1/ \
    --kg_path data/freebase_lmdb/ \
    --output_dir checkpoints/stage2_rlmc/ \
    --epochs 10 \
    --lr 1e-4
```

*A2C policy gradient over frozen Stage I. Action space: {Tight, Medium, Loose, Stop}.*

### Stage II — STRL Variant (PPO + InfoNCE curriculum)

```bash
python train/exp15_strl.py \
    --planner_ckpt checkpoints/stage1/ \
    --kg_path data/freebase_lmdb/ \
    --output_dir checkpoints/stage2_strl/ \
    --epochs 20 \
    --ppo_clip 0.2 \
    --infonce_temp 0.07
```

*Jointly fine-tunes Stage I + II with InfoNCE semantic teacher (w_sem: 1.0→0.3 curriculum).*

### Stage III — CDS Pipeline

```bash
# 1. Collect traversal candidates
python train/exp16_collect_data.py --rl_ckpt checkpoints/stage2_rlmc/

# 2. Train F1 bi-encoder (MiniLM)
python train/exp16_cds.py --stage f1

# 3. Train F2 path ranker (MPNet)
python train/exp27_train_s2_mpnet.py

# 4. Build DPO preference dataset
python train/exp37_build_dpo_dataset.py

# 5. SFT reference model (Flan-T5)
python train/exp26_t5_generative_s3.py

# 6. DPO judge (F3)
python train/exp38_train_t5_dpo.py \
    --sft_ckpt checkpoints/stage3_sft/ \
    --output_dir checkpoints/stage3_dpo/ \
    --beta 0.1 --lr 1e-6 --epochs 2
```

---

## Hyperparameters

| Component | Key Hyperparameters |
|---|---|
| **Stage I (Planner)** | AdamW lr=1e-5, 30 epochs, batch=16 (4×4 accum), FP16 |
| **Stage II RLMC (A2C)** | lr=1e-4, γ=0.99, 10 epochs, batch=8 |
| **Stage II STRL (PPO)** | ε=0.2, γ=0.99, λ_GAE=0.95†, τ=0.07, 20 epochs |
| **F1 Bi-encoder** | MiniLM-L6-v2, lr=2e-5, 5 epochs, top-200 |
| **F2 Path Ranker** | MPNet-base-v2, lr=2e-5, 5 epochs, top-50 |
| **F3 DPO Judge** | Flan-T5-Base, lr=1e-6, β=0.1, 2 epochs |

†λ_GAE=0.95 is the standard default from [Schulman et al., 2016 (GAE)](https://arxiv.org/abs/1506.02438).

---

## Evaluation

```bash
# End-to-end evaluation (Hit@1 + Reasoning Recall)
python eval/e2e_evaluate.py \
    --planner_ckpt checkpoints/stage1/ \
    --rl_ckpt checkpoints/stage2_rlmc/ \
    --cds_dir checkpoints/stage3_dpo/ \
    --data_dir data/cwq/ \
    --split dev

# Extended metrics (latency, VRAM, path accuracy, depth accuracy)
python eval/extended_metrics_eval.py

# Verify paper numbers
python scripts/verify_paper_numbers.py
```

---

## Key Design Decisions

**Why 4 actions instead of raw relation IDs?**
Freebase has >10,000 relations. Direct RL over relation IDs leads to policy collapse from sparse rewards and an intractably large action space. Our 4 high-level width-control actions keep the policy head small (~4K parameters) and optimization stable.

**Why decouple Stage I and Stage II training?**
Joint optimization causes the RL signal to corrupt the planner's cross-hop attention before stable representations form, leading to policy divergence within 3 epochs. Freezing Stage I during RL training provides a stable semantic manifold for policy learning.

**Why DPO for Stage III?**
DPO directly instantiates RL-family preference alignment at the candidate selection stage — making Stage III a second RL-layer consistent with the paper's narrative. It also avoids training an explicit reward model, reducing Stage III training complexity.

**RLMC vs STRL trade-off:**
STRL's InfoNCE curriculum tightly grounds semantic representations, reducing traversal to ~14 edges/question. However, its tighter beam produces noisier candidate sets at CDS, slightly reducing Hit@1 (40.4%) vs RLMC (42.1%). The two variants demonstrate a precision-recall trade-off rather than one strictly dominating.

---

## Paper

The full paper LaTeX source is in [`paper_kgqarl/`](paper_kgqarl/):

```
paper_kgqarl/
├── main.tex              # Main paper (IEEE conference format)
├── references.bib        # Bibliography
└── paper/tikz/
    ├── fig_detailed_dataflow.tex   # Architecture diagram (Fig. 2)
    ├── fig_kg.tex                  # KG reasoning trace (Fig. 1)
    └── fig_working_example.tex     # Inference trace (Appendix A)
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
