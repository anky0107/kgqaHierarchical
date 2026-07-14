# Paper Code

This folder contains **only the code directly used to produce the results in the paper**.
Each file is named to match its corresponding paper section rather than internal experiment numbering.

```
paper_code/
+-- stage1_planner/
¦   +-- train_planner.py          # §V-A  Stage I: Semantic Reasoning Planner
¦                                 #        RoBERTa-Large + Relation/Domain/Confidence/Stop heads
¦                                 #        Multi-task supervised training (30 epochs, AdamW lr=1e-5)
¦
+-- stage2_rl/
¦   +-- train_rlmc.py             # §V-B  Stage II: RLMC — A2C Traversal Controller
¦   ¦                             #        4-action policy {Tight, Medium, Loose, Stop} over frozen Stage I
¦   ¦                             #        GAE advantage, lr=1e-4, 10 epochs
¦   +-- train_strl.py             # §V-C  STRL variant — PPO + InfoNCE curriculum
¦                                 #        Joint Stage I+II fine-tuning, w_sem: 1.0->0.3, 20 epochs
¦
+-- stage3_cds/
¦   +-- collect_candidates.py     # Collect traversal candidates from Stage II for CDS training
¦   +-- precompute_entity_embs.py # Pre-compute MiniLM-L6 entity embeddings (F1 Fast Filter)
¦   +-- train_f2_path_ranker.py   # §V-D  F2: MPNet path-aware ranker -> top-50 (lr=2e-5, 5 ep)
¦   +-- train_f3_sft.py           # §V-D  F3: Flan-T5-Base SFT reference model (lr=3e-5, 3 ep)
¦   +-- build_dpo_dataset.py      # Build preference pairs (gold chosen, top incorrect rejected)
¦   +-- train_f3_dpo.py           # §V-D  F3: Flan-T5 DPO Judge (beta=0.1, lr=1e-6, 2 epochs)
¦
+-- eval/
¦   +-- evaluate_e2e.py           # End-to-end: Reasoning Recall + Hit@1  (Table 2)
¦   +-- evaluate_extended_metrics.py  # Full Table 3: Hits@5, Path Acc, Depth Acc, Latency, VRAM
¦   +-- evaluate_hit1.py          # Standalone Hit@1 evaluation
¦
+-- shared/
¦   +-- kg_loader.py              # Freebase LMDB graph loader (shared across all stages)
¦
+-- scripts/
    +-- filter_freebase.py        # Build Freebase subgraph from raw RDF dump -> LMDB
    +-- gen_selector_data.py      # Generate CDS (Stage III) training data
    +-- verify_paper_numbers.py   # Reproduce exact numbers from Tables 2 & 3
```

## Running Order

```
1.  scripts/filter_freebase.py           # One-time: build LMDB KG from Freebase RDF
2.  stage1_planner/train_planner.py      # Train Stage I (Semantic Planner)
3a. stage2_rl/train_rlmc.py              # Train Stage II — RLMC variant
3b. stage2_rl/train_strl.py              # Train Stage II — STRL variant (alternative)
4.  stage3_cds/collect_candidates.py     # Collect Stage II traversal outputs
5.  stage3_cds/precompute_entity_embs.py # Pre-compute F1 entity embeddings
6.  stage3_cds/train_f2_path_ranker.py   # Train F2 path ranker
7.  stage3_cds/train_f3_sft.py           # Train F3 SFT reference model
8.  stage3_cds/build_dpo_dataset.py      # Build DPO preference dataset
9.  stage3_cds/train_f3_dpo.py           # Train F3 DPO judge
10. eval/evaluate_e2e.py                 # Full pipeline evaluation
```

See the root [README.md](../README.md) for full setup, data prep, and hyperparameter details.
