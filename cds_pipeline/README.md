# CDS Pipeline

Clean, bug-fixed inference package for the Cascading Dust Separator.

## Structure

```
cds_pipeline/
├── __init__.py    — package entry point
├── models.py      — PathAwareRanker + load_stage1/2/3 functions
├── pipeline.py    — CDSPipeline: the 3-stage cascade
├── utils.py       — path flattening, relation→NL helpers
└── evaluate.py    — benchmark script (Hit@1/3/10 on dev JSON)
```

## Bugs Fixed vs Original Benchmark Scripts

| Bug | Location | Fix |
|---|---|---|
| Stage 2 bypasses trained MLP | `exp16v2_benchmark.py` L68 | `pipeline.py` calls `PathAwareRanker.forward()` |
| Path serialised as `list[list[str]]` → garbage string | all benchmark scripts | `utils.flatten_path()` extracts first relation per hop |
| Stage 2 uses MiniLM tokenizer instead of MPNet | `benchmark_final_cds_v3.py` L96–98 | `pipeline.py` uses `self.s2_tok` (MPNet) throughout |
| S1 hard-capped at top-100 | all benchmark scripts | Adaptive `min(200, N)` |

## Usage

```bash
# Evaluate v2 Stage 3 (name-only)
python -m cds_pipeline.evaluate --s3 v2

# Evaluate v3 Stage 3 (path-aware)
python -m cds_pipeline.evaluate --s3 v3

# Compare both in one run
python -m cds_pipeline.evaluate --compare
```

## Programmatic usage

```python
from cds_pipeline.pipeline import CDSPipeline

pipe = CDSPipeline(s3_version="v3")

ranked = pipe.rank(
    question   = "Who directed Saving Private Ryan?",
    candidates = [{"name": "Steven Spielberg", "is_gold": True}, ...],
    path       = [["film.film.directed_by"]],  # list[list[str]] or str
)
# ranked[0] is the predicted answer
```

## Checkpoints Required

| Stage | Checkpoint |
|---|---|
| S1 | `checkpoints/exp16v2_s1_bi.pt` |
| S2 | `checkpoints/exp16v2_s2_path.pt` |
| S3 v2 | `checkpoints/exp16v2_s3_cross.pt` |
| S3 v3 | `checkpoints/exp16v3_s3_cross.pt` |
