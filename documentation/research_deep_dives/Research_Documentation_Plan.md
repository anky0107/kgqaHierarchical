# Implementation Plan: Comprehensive Exp 10 Technical Specification

The objective is to expand the existing Experiment 10 documentation into a "Thesis-Level" technical manual. This will address the missing details regarding the data pipeline, the SPARQL parsing mechanism, and the Knowledge Graph simulation strategy (Gold Subgraph approach).

## User Review Required

> [!IMPORTANT]
> The documentation will explicitly clarify that we do **not** use a full 400GB Freebase instance. Instead, we use a **Local Gold Subgraph** strategy. Please confirm if you want me to include the mathematical proof of why this is equivalent for the test set.

## Proposed Changes

### [Component] Documentation Expansion
#### [MODIFY] [Exp10_Universal_Planner_High_Fidelity.md](file:///documentation/research_deep_dives/Exp10_Universal_Planner_High_Fidelity.md)
- **Dataset Orchestration**:
    - Detail the `UniversalDataset` class and the `[TAG] topic: ...` prefixing logic.
    - Deep dive into `WeightedRandomSampler` and its role in preventing catastrophic forgetting.
- **Knowledge Graph Simulation (The "Local KB")**:
    - Explain the hardware constraints of Full Freebase.
    - Detail the **Local Gold Subgraph** strategy: harvesting 100% of triples from train/dev/test SPARQL queries to build a 0.5GB in-memory graph.
- **SPARQL Parser Deep Dive**:
    - Explain the `extract_triples` regex logic (including the inverse `^` operator fix).
    - Diagram the flow from `Raw SPARQL` -> `Triple Extractor` -> `Adjacency List`.
- **Multi-Dataset Benchmarks**:
    - Formalize Hits@1 for WebQSP and MetaQA.
- **Visual Spec (Mermaid Diagrams)**:
    - **Diagram A**: Inference Pipeline (Question to Answer via BFS).
    - **Diagram B**: KB Extraction Pipeline (Dataset to Graph).
    - **Diagram C**: Joint Training Architecture.

## Verification Plan

### Manual Verification
- Cross-reference all Mermaid diagrams with the actual logic in `utils/sparql_parser.py` and `eval/execution_eval_all.py`.
- Ensure all file links are absolute and clickable.
- Confirm that the Freebase load constraint explanation aligns with the `build_kg_from_cwq_triples` implementation.
