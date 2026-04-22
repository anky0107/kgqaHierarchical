# Implementation Plan: Formalizing Experiment 10 Research Paper

The objective is to produce a high-fidelity, comprehensive research paper for **Experiment 10 (Universal Planner)** that matches the granular detail of the Experiment 9 paper. This includes covering the architectural fusion, joint training logic, and "Blind Evaluation" benchmarks.

## Proposed Changes

### [Component] Research Documentation
#### [NEW] [exp10_research_paper.md](documentation/research_deep_dives/exp10_research_paper.md) (Rewrite)
- **Granular Architecture**: Detail the `UniversalPlanner` class, specifically the additive fusion of `dataset_embedding` with the RoBERTa [CLS] token.
- **Training Logic**: Breakdown of the `UniversalDataset` construction and the `WeightedRandomSampler` logic for balancing CWQ/WebQSP/MetaQA.
- **Tiered Optimization**: Detailed explanation of the joint loss function ($Loss_{total} = Loss_{domain} + Loss_{relation} + Loss_{stop}$) and the 8-step gradient accumulation strategy.
- **Block Diagrams**: High-fidelity Mermaid diagrams for both the **Inference Flow** and the **Joint Training Pipeline**.
- **Execution Benchmarks**: Formalize the 70.93% (Tagged) and 68.41% (Blind) Results on the CWQ Test Set.

## Verification Plan

### Manual Verification
- Review the logic breakdown against [exp10_universal.py](file:///train/exp10_universal.py) to ensure 100% accuracy in code references.
- Confirm Mermaid diagrams render correctly.
