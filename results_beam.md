# KGQA Research Experiment Results

## End-to-End Evaluation

Evaluation protocol: question â†’ model predicts relation path â†’ path match against gold SPARQL â†’ derive answer correctness.
This matches the planning evaluation used by DRKG, DAMR, and Plan-Then-Retrieve.

| Model | Hits@1 | Hits@3 | Hop Accuracy | Questions |
|---|---|---|---|---|
| **Exp 7-Beam (Dev)** | 0.4076 | 0.6773 | 0.5861 | 3496 |
| **Exp 7-Beam (Test)** | 0.4294 | 0.6598 | 0.5960 | 3498 |

### Breakdown by Number of Hops

| Model | 1-hop | 2-hop | 3-hop | 4-hop |
|---|---|---|---|---|
| **Exp 7-Beam (Dev)** | 0.3304 (1141) | 0.4406 (1625) | 0.4246 (570) | 0.5625 (160) |
| **Exp 7-Beam (Test)** | 0.3458 (1177) | 0.4557 (1670) | 0.5173 (462) | 0.5080 (187) |

---

## Comparable Published Results on CWQ

| Method | Hits@1 | F1 | Year |
|---|---|---|---|
| NSM | 0.486 | 0.483 | 2021 |
| SR+NSM | 0.505 | - | 2022 |
| TIARA | 0.534 | - | 2022 |
| ChatKBQA | 0.555 | - | 2024 |
| DRKG | 0.669 | - | 2025 |

> **Note**: Our evaluation uses path-matching on a CWQ-derived subgraph.
> Published results use Freebase execution. Direct comparison should be interpreted carefully.
> Our Hits@1 measures *planning accuracy* (does the model predict the correct relation path?)
> which upper-bounds the final answer accuracy.

---

## Performance Notes

- **GPU**: RTX 5070 Laptop (SM 12.0 / Blackwell)
- **PyTorch**: 2.11.0+cu128 with Mixed Precision (AMP)
- **Dataset**: ComplexWebQuestions (CWQ) 1.1
- **Evaluation**: Path-match based (planning accuracy)
