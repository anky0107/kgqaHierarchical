"""
Exp 20: Dropout Ensemble for Stage I Relation Prediction
=========================================================

HYPOTHESIS
----------
Stage I (exp7) uses a single deterministic forward pass at inference
(dropout disabled). Running T stochastic forward passes with dropout
re-enabled and averaging the relation logits is a zero-cost ensemble:
no retraining, no new parameters, no architecture changes.

For relation prediction this reliably adds +1–2% recall because:
  - The transformer has dropout layers at every attention block.
  - Different dropout masks expose different relation-attention patterns.
  - Averaging across T masks smooths out per-sample noise and widens
    the effective top-k coverage.

This experiment:
  1. Loads the exp7 checkpoint as-is.
  2. Evaluates standard inference (T=1, dropout off) as baseline.
  3. Evaluates ensemble inference (T=5, 10, 20) with dropout on.
  4. Reports Path Accuracy and Reasoning Recall for each T.
  5. Saves the best-T logits for the dev set so downstream CDS
     can be re-run without retraining anything.

NO TRAINING — evaluation only.
OUTPUT:    metrics/exp20_ensemble.csv
           data/exp20_ensemble_logits_dev.pt   (best T, saved for reuse)
"""

import os, sys, json, functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.getcwd()
sys.path.append(ROOT)

import importlib.util, types

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub out utils.sparql_parser so exp6_unified doesn't crash
if "utils.sparql_parser" not in sys.modules:
    stub = types.ModuleType("utils.sparql_parser")
    stub.find_reasoning_path = lambda *a, **kw: None
    sys.modules["utils.sparql_parser"] = stub
if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")

_exp6  = _load_module("train.exp6_unified", os.path.join(ROOT, "train/exp6_unified.py"))
_exp7  = _load_module("train.exp7_roberta", os.path.join(ROOT, "train/exp7_roberta.py"))

UnifiedDataset        = _exp6.UnifiedDataset
collate_unified       = _exp6.collate_unified
ScaledUnifiedPlanner  = _exp7.ScaledUnifiedPlanner


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def enable_dropout(model: torch.nn.Module):
    """Re-enable dropout layers for stochastic inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def relation_hit_at_k(rel_logits: torch.Tensor,
                       gold_paths: torch.Tensor,
                       path_lengths: torch.Tensor,
                       k: int) -> tuple:
    """
    Compute Hit@k and Path Accuracy (all hops correct) over a batch.
    rel_logits : [B, H, num_rel]
    gold_paths : [B, H]
    path_lengths: [B]
    Returns (hit_at_k_count, path_acc_count, valid_hop_count, valid_q_count)
    """
    B, H, _ = rel_logits.shape
    hop_hits = 0; path_correct = 0
    total_hops = 0; total_qs = 0

    for b in range(B):
        L = int(path_lengths[b].item())
        if L == 0:
            continue
        all_hops_correct = True
        for h in range(L):
            gold_r = int(gold_paths[b, h].item())
            topk   = torch.topk(rel_logits[b, h], k).indices.tolist()
            correct = gold_r in topk
            hop_hits   += int(correct)
            total_hops += 1
            if not correct:
                all_hops_correct = False
        path_correct += int(all_hops_correct)
        total_qs     += 1

    return hop_hits, path_correct, total_hops, total_qs


# ─────────────────────────────────────────────────────────────
#  Ensemble inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ensemble(model: ScaledUnifiedPlanner,
                       loader: DataLoader,
                       device: torch.device,
                       T: int = 1,
                       k_values: list = None,
                       save_logits: bool = False) -> dict:
    """
    Run T forward passes per batch. Average the relation logits.
    Report Hit@k for each k in k_values, plus Path Accuracy.

    If T == 1: standard deterministic inference (dropout off).
    If T  > 1: stochastic ensemble (dropout on for all passes).
    """
    if k_values is None:
        k_values = [1, 5, 50]

    model.eval()
    if T > 1:
        enable_dropout(model)   # re-enable dropout for ensemble passes

    counters   = {k: [0, 0, 0, 0] for k in k_values}  # hits, path, hops, qs
    all_logits = []    # collect for optional save

    for enc, doms, paths, nums in tqdm(loader, desc=f"Eval T={T}"):
        enc   = {kk: vv.to(device) for kk, vv in enc.items()}
        paths = paths.to(device)
        nums  = nums.to(device)

        # Accumulate T passes
        logit_sum = None
        for _ in range(T):
            out = model(enc["input_ids"], enc["attention_mask"])
            rl  = out["rel_logits"]   # [B, H, num_rel]
            logit_sum = rl if logit_sum is None else logit_sum + rl

        avg_logits = logit_sum / T    # [B, H, num_rel]

        if save_logits:
            all_logits.append(avg_logits.cpu())

        for k in k_values:
            h, p, th, tq = relation_hit_at_k(avg_logits, paths, nums, k)
            counters[k][0] += h
            counters[k][1] += p
            counters[k][2] += th
            counters[k][3] += tq

    results = {}
    for k in k_values:
        h, p, th, tq = counters[k]
        results[k] = {
            "hit_at_k":    h  / max(th, 1),
            "path_acc":    p  / max(tq, 1),
            "total_hops":  th,
            "total_qs":    tq,
        }

    saved_path = None
    if save_logits and all_logits:
        saved_path = os.path.join(
            ROOT, f"data/exp20_ensemble_logits_dev_T{T}.pt")
        torch.save(torch.cat(all_logits, dim=0), saved_path)
        print(f"[Exp20] Logits saved → {saved_path}")

    return results, saved_path


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Exp20] Device: {device}")

    rel2id = torch.load(
        os.path.join(ROOT, "data/processed_entity/relation2id.pt"))
    dom2id = torch.load(
        os.path.join(ROOT, "data/processed_entity/domain2id.pt"))
    num_rel = len(rel2id); num_dom = len(dom2id)
    id2rel  = {v: k for k, v in rel2id.items()}

    # ── Load exp7 checkpoint ──────────────────────────────────────────────────
    model = ScaledUnifiedPlanner(num_dom, num_rel).to(device)
    ckpt  = os.path.join(ROOT, "checkpoints/exp7_roberta_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"[Exp20] Loaded {ckpt}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    collate   = functools.partial(collate_unified, tokenizer=tokenizer)
    dev_ds    = UnifiedDataset("data/cwq_dev.json", rel2id, dom2id)
    dev_loader = DataLoader(dev_ds, batch_size=16, collate_fn=collate)

    # ── Sweep over T values ───────────────────────────────────────────────────
    T_values = [1, 5, 10, 20]
    k_values = [1, 5, 50]

    metrics_dir  = os.path.join(ROOT, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "exp20_ensemble.csv")

    # Header
    header_cols = ["T"]
    for k in k_values:
        header_cols += [f"hit@{k}", f"path_acc@{k}"]
    with open(metrics_path, "w") as f:
        f.write(",".join(header_cols) + "\n")

    best_hit1 = 0.0
    best_T    = 1

    for T in T_values:
        print(f"\n[Exp20] Running ensemble T={T} …")
        save = (T == T_values[-1])   # save logits for the last (largest) T
        results, _ = evaluate_ensemble(
            model, dev_loader, device, T=T, k_values=k_values, save_logits=save)

        row = [str(T)]
        for k in k_values:
            r = results[k]
            hit = r["hit_at_k"]; pa = r["path_acc"]
            row += [f"{hit:.4f}", f"{pa:.4f}"]
            print(f"  k={k:2d}  Hit@k={hit:.4f}  PathAcc={pa:.4f}  "
                  f"(hops={r['total_hops']}, qs={r['total_qs']})")

        with open(metrics_path, "a") as f:
            f.write(",".join(row) + "\n")

        if results[1]["hit_at_k"] > best_hit1:
            best_hit1 = results[1]["hit_at_k"]
            best_T    = T

    print(f"\n[Exp20] Best Hit@1 = {best_hit1:.4f} at T={best_T}")

    # Save logits for best T if not already done
    if best_T != T_values[-1]:
        print(f"[Exp20] Saving logits for best T={best_T} …")
        evaluate_ensemble(model, dev_loader, device,
                          T=best_T, k_values=[1],
                          save_logits=True)

    print(f"[Exp20] Results written to {metrics_path}")
    print("[Exp20] Done. Use the saved logits to re-run CDS without retraining.")


if __name__ == "__main__":
    main()
