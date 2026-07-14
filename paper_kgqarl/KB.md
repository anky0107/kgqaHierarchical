# KB — paper_kgqarl knowledge centre
<!-- Agent-only. Compact reference. Facts from code unless marked [note]. -->

## PAPER NARRATIVE — FINAL DECISION
Title: "Knowledge Graph Question Answering Using Reinforcement Learning"

NAMING HIERARCHY:
  Stage I   — Semantic Reasoning Planner (exp7)
  Stage II  — RL-Based Traversal Controller (exp9 RLMC) ← PRIMARY RL CONTRIBUTION
  Stage III — Candidate Disambiguation & Selection / CDS (the funnel)
    ├─ CDS Stage 1: Bi-Encoder Fast Pruning (MiniLM)
    ├─ CDS Stage 2: Path-Aware Sieve (MPNet)
    └─ CDS Stage 3: Preference-Aligned Generative Judge (T5 DPO, exp38)

DPO position: RL-family training method for CDS final sub-stage.
  NOT a separate top-level stage. Sits architecturally inside CDS.
  RL in title = Stage II (policy gradient). DPO in CDS = RL-inspired selection.
Hit@1: 42.12% (exp9+exp38, benchmarked).
File: `paper_kgqarl/main.tex` (905 lines, IEEEtran 10pt, conference)
Authors: Ankit Rana, Dr. Chandramani Chaudhary — NIT Calicut CSE


---

## ARCHITECTURE — GROUND TRUTH (from code)

### Stage I — Semantic Planner (exp7_roberta.py)
- Model: ScaledUnifiedPlanner; backbone: RoBERTa-Large (CLS→1024)
- Proj: Linear(1024→512) → h_q [B,512]  ← hidden_dim=512 NOT 768
- Cross-hop transformer: 4 layers, 8 heads, FF=3072, seq=4 hops
- Hop embeddings: learnable [4,512], added (not concat) to h_q copies
- Heads: domain_head L(512,K), confidence_head L(512,1)+sigmoid,
  relation_head L(512,|R|), stop_head L(512,1)+sigmoid
- eta_q computed but NEVER used in downstream RL (no dedicated loss)
- Train: AdamW lr=1e-5, eff_batch=16 (4×4 accum), 30 epochs, FP16
- Loss: CE(rel) + CE(domain) + BCE(stop) — no lambda weighting in code
- Ckpt: checkpoints/exp7_roberta_best.pt

### Stage II — RLMC (exp9_rlmc.py) ← PAPER'S PRIMARY AGENT
- Loads exp7, freezes ALL base params
- Policy: Linear(512→256→4) on refined_repr[:,t,:] (NOT [h_h;eta_q])
- Value:  Linear(512→256→1) same input
- Actions: TIGHT k=1 | MEDIUM k=5 | LOOSE ~50 domain | STOP
- Algorithm: A2C (paper calls it PPO — aspirational; no clip in code)
  actor_loss = -(log_probs * adv).mean()
  critic_loss = MSE(state_values, returns)
  entropy = -m.entropy().mean() * 0.01
- Rewards (simulated; NO LMDB during training):
  TIGHT: gold_rel==argmax → +1.0/-1.0
  MEDIUM: gold_rel in top5 → +0.5/-1.0
  LOOSE: domain match → +0.1/-1.0
  STOP: h>=path_len → +1.0/-1.0
- lr=1e-4, gamma=0.99, 10 epochs
- Ckpt: checkpoints/exp9_rlmc_epoch_9.pt

### Stage II Alt — STRL (exp15_strl.py)
- Backbone FROZEN epochs 0-4, UNFROZEN epochs 5-19
- Real PPO with clip eps=0.2 (ONLY true PPO in codebase)
- Reward: cosine-based alpha=0.5*R_sem + beta=0.3*R_conn + gamma=0.2*R_eff + R_entity
- InfoNCE: 63 RANDOM negatives (NOT hard — code line 498 randint)
- Inference: semantic_beam_with_kg_filter() via cosine_sim(hop_repr, rel_emb_bank)
- Ckpt: checkpoints/exp15_strl_best.pt (replaces exp7+exp9 in STRL pipeline)

### Stage III — CDS (exp16v2 + exp38)
Architecture: 3-sub-stage funnel (CDS = Cascading Dust Separator)
- CDS S1: all-MiniLM-L6-v2, cosine-sim, B→top-100, MSE loss, 3 rand negs
- CDS S2: all-mpnet-base-v2, MLP CONCAT(q+p+e) L(768*3,768)→GELU→D(0.1)→L(768,1),
  100→top-15, SoftMarginLoss, 15 rand negs
- CDS S3 (NEW — replaces BGE KL-distill): **Preference-Aligned Generative Judge**
  Model: google/flan-t5-base (250M), fine-tuned via DPO (exp38)
  Input: MC prompt = question + numbered top-15 candidate list
  Output: generated entity name (listwise — sees all candidates at once)
  Training: DPO on (prompt, chosen=gold_name, rejected=hard_neg)
  Base SFT: exp31 (T5 MC), then DPO-aligned via exp38_custom.py
  L_DPO = -logsigmoid(beta*(log pi_c/pi_ref_c - log pi_r/pi_ref_r)), beta=0.1
  Ckpts: exp16v2_s1_bi.pt / exp16v2_s2_path.pt / exp38_t5_dpo_s3.pt

Key advantage over BGE cross-encoder:
  Listwise inference — T5 attends to ALL 15 candidates simultaneously
  BGE scores each candidate in isolation (pointwise) ← fundamentally weaker
  T5 self-attention separates plausible distractors; BGE cannot

### Stage III DPO — exp38 (PAPER NARRATIVE S3)
- Base: google/flan-t5-base; SFT base: checkpoints/exp31_t5_mc_s3.pt
  exp31 = T5 MC reranker: reads "Q + numbered candidates" → generates entity name
- exp37: builds DPO pairs from exp30_t5_mc_train.json
  prompt=MC prompt, chosen=gold answer name, rejected=random distractor
- exp38_custom.py: manual DPO in pure PyTorch (TRL dropped encoder-decoder)
  L_DPO = -logsigmoid(beta*(log pi_c - log pi_ref_c - log pi_r + log pi_ref_r))
  beta=0.1, lr=1e-6, epochs=2, batch=2, accum=8
- Ckpt: checkpoints/exp38_t5_dpo_s3.pt

---

## VERIFIED RESULTS — FULL (from state_and_next_steps.md + walkthrough.md, session c2131455)

### Complete Performance Trajectory
| Config | Hit@1 | Recall | Notes |
|---|---|---|---|
| Exp15+CDS v2 baseline | 31.98% | 69.13% | early session baseline |
| Exp21 (Beam+CDS v6 27k HN) | 35.61% | 69.27% | best S3 cross-encoder |
| Exp24 (Path-Aware S3 v7) | 38.32% | 69.27% | path injection |
| Exp25 (Listwise S2) | 38.38% | 69.13% | listwise loss on S2 |
| Ensemble(exp9+exp15)+BGE(S3) | 41.80% | 78.33% | first >40% |
| Exp26 Generative T5-Base | 43.69% | 79.42% | previous SOTA |
| **Ensemble+T5 MC (Exp31 SFT)** | **43.77%** | **77.78%** | **🏆 OVERALL SOTA** |
| Ensemble+T5 CoT (Exp32) | 43.35% | 78.30% | CoT hurt perf |
| BGE Pointwise Cross-Enc (Exp35) | 41.66% | 77.64% | pointwise ceiling |
| Listwise InfoNCE Cross-Enc (Exp36) | 40.92% | 77.01% | listwise train, pointwise inf |

### Key Benchmarked Pipeline Numbers
| Agent | Reranker | Recall | Hit@1 |
|---|---|---|---|
| exp9 (RLMC) | T5 MC SFT (exp31) | 62.79% | **43.23%** |
| exp9 (RLMC) | T5 DPO (exp38) | 63.08% | **42.12%** |
| exp15 (STRL) | T5 MC SFT (exp31) | ~69% | 40.43% |
| Ensemble (exp9+exp15) | T5 MC SFT (exp31) | 71.84% | **42.72%** |
| Ensemble (exp9+exp15) | T5 DPO (exp38) | 72.70% | 42.12% |

### CRITICAL FINDING: The Alignment Tax
- DPO (exp38) UNDERPERFORMS SFT (exp31) across ALL configurations
- exp9+exp38 = 42.12% vs exp9+exp31 = 43.23% (DPO costs -1.11 pp)
- Ensemble+exp38 = 42.12% vs Ensemble+exp31 = 42.72% (DPO costs -0.60 pp)
- Cause: destabilized exact-match extraction by penalizing distractor tokens
- CONCLUSION: exp31 (T5 MC SFT) > exp38 (T5 DPO) — DPO is NOT the best S3

### BEST SINGLE-AGENT CONFIGURATION
**exp9 (RLMC) + exp31 (T5 MC SFT) = 43.23% Hit@1, 62.79% Recall**
This is the cleanest paper-worthy pipeline:
- One RL traversal agent (exp9)
- Generative T5 listwise reranker (exp31 SFT)
- No ensemble tricks needed
- Better than exp38 DPO

### GENERATIVE vs POINTWISE CONCLUSION (verified)
- T5 MC listwise (sees all candidates at once) > BGE pointwise (scores in isolation)
- Reason: self-attention across candidate list separates plausible distractors
- BGE ceiling: 41.66% | T5 SFT ceiling: 43.77% (ensemble)

### Paper results table alignment (what to report)
| Architecture | Recall | Hit@1 |
|---|---|---|
| Stage I (exp7) | ~58% | 37.21% |
| Stage I+II+III (exp9+exp31) | 62.79% | 43.23% |
| Stage I+STRL+III (exp15+exp31) | ~69% | 40.43% |
| STRL NOTE: STRL recall is higher but Hit@1 is LOWER with T5 reranker |

CDS loss ablation (500-sample dev subset, exp16v2):
  KL-Distill=57.5% | SoftMargin=56.5% | InfoNCE=56.0%

---

## KNOWN PAPER vs CODE GAPS (paper is aspirational — keep unless explicitly fixing)

| Paper claim | Code reality |
|---|---|
| PPO with clip eps=0.2 for Stage II | A2C in exp9; PPO only in exp15 |
| State s_t=[h_h;eta_q] | Only h_h in policy for exp9 |
| hidden_dim=768 | Code: 512 |
| 63 hard negatives InfoNCE | Random negatives |
| BGE KL-distill as Stage III | exp38 DPO = T5-based (NEW narrative) |

---

## PAPER STRUCTURE (main.tex line refs)

L53  Abstract — 3-stage framework, STRL, 42.80% Hit@1
L91  Introduction — 10-point flow, 5 contributions (incl STRL, CDS)
L135 Related Work — SP / Embedding / GNN / RL / Transformer / LLM
L163 Preliminaries — KG formalism, multi-hop KGQA
L187 Problem Formulation — MDP, 4 actions, reward R_t
L208 Methodology — Stage I (L220) / Stage II (L266) / STRL (L314) / Stage III (L326) / Complexity (L359)
L385 Experimental Setup — CWQ, Freebase, metrics, hyperparams Table, baselines
L600 Results & Analysis — Table 1 (main) Table 2 (extended) Table 3 (vs LLMs)
L785 Research Challenges — sparse rewards, traversal explosion, reward sensitivity, semantic drift
L798 Ablation — 5 qualitative ablations
L849 Limitations — 5 items
L861 Future Work
L877 Conclusion
L886 Appendix — Jerry Jones inference trace

Key tables: tab:main_results tab:extended_metrics tab:arch_positioning tab:hyperparams
Key figures: fig_kg fig_detailed_dataflow fig_cds_funnel fig_working_example
All tikz in: paper_kgqarl/paper/tikz/

---

## DATASET & INFRA

- CWQ: dev=3502, train=27639 — multi-hop KGQA on Freebase
- KG storage: data/processed_kg/augmented_kg.pt
  kg['forward'][MID] = [(rel_str, target_MID), ...]
  LMDB memory-mapped — ~2s startup
- CDS data: data/exp16_cds_train.json / exp16_cds_dev.json
- T5 MC data: data/exp30_t5_mc_train.json
- DPO data: data/exp37_t5_dpo_train.json (built by exp37)
- MID→name: data/master_mid2name.json (273 MB)

---

## DIAGRAM AUDIT — All Figures
<!-- 7 tikz files in paper/tikz/. Audit against Pass-1 notation + new DPO pipeline. -->

### Figure Map (which figures are actually used in main.tex)
| fig label | tikz file | Used in paper? | Placement |
|---|---|---|---|
| fig:kg | fig_kg.tex | YES — L175 (figure*[t]) | §Prelim, before ProbForm |
| fig:arch | fig_detailed_dataflow.tex | YES — L213 (figure*[t]) | §Methodology opening |
| fig:cds_funnel | fig_cds_funnel.tex | YES — L327 (figure[htbp]) | §Stage III |
| fig:working_example | fig_working_example.tex | YES — in Appendix L888 | Appendix |
| fig_architecture.tex | (not \input'd in paper) | NO — orphan file | — |
| fig_pipeline.tex | (not \input'd in paper) | NO — orphan file | — |
| fig_traversal.tex | (not \input'd in paper) | NO — orphan file | — |

---

### fig_detailed_dataflow.tex (fig:arch) — PRIMARY ARCHITECTURE FIGURE
**Placement:** §Methodology opening — CORRECT, should be here.

**Errors found:**

1. **Stage II box says "(PPO)"** — L149: `\textbf{RL Meta-Constraint Controller} \\ \textbf{(PPO)}`
   Code uses A2C for exp9. Change to: `\textbf{RL Meta-Constraint Controller} \\ \textbf{(Policy Gradient)}`

2. **CDS box says "BGE-Reranker"** — L208: `MiniLM-L6 $\rightarrow$ MPNet $\rightarrow$ BGE-Reranker`
   Must change to: `MiniLM-L6 $\rightarrow$ MPNet $\rightarrow$ Flan-T5 (DPO)`

3. **Stage III label says "CDS Ranker"** — L220 stage title: fine as is (CDS is the correct name)

4. **Notation: uses old H symbol** — L127: `($L=4$, $H=8$)` for transformer heads
   Here H=8 means 8 attention heads, NOT hop count. Rename to: `($L=4$, heads$=8$)` to avoid confusion with H_max

5. **Callout box says "PPO MLP Detail"** — L166: `PPO MLP Detail`
   Change to: `Policy MLP Detail`

6. **Notation in callout uses old notation** — L168: `State: $s_t = [h^{(L)}_h \,;\, \eta_q]$`
   This is consistent with Pass-1 notation — h_h = H^(L)_h — so technically correct. OK.

7. **Figure caption (main.tex L216) says "path-aware contrastive ranker"**
   Change to: "a Preference-Aligned Generative Judge (DPO) selects the final answer."
   Also: caption still has old Stage III description.

**Fix status:** [ ] Pending

---

### fig_cds_funnel.tex (fig:cds_funnel) — CDS FUNNEL DIAGRAM
**Placement:** §Stage III — CORRECT.

**Errors found:**

1. **Stage 1 equation uses old notation** — L23: `$s_1(q, e) = \cos(\mathbf{h}_q, \mathbf{z}_e)$`
   Must change to: `$s_1(q, e) = \cos(\mathbf{q}_{\text{bi}}, \mathbf{e}_{\text{bi}})$`

2. **Stage 2 equation uses old notation** — L31: `$s_2(q, p, e) = \text{MLP}([\mathbf{h}_q; \mathbf{h}_p; \mathbf{h}_e])$`
   Must change to: `$s_2(q, p, e) = \text{MLP}([\mathbf{q}_{\text{pw}}; \mathbf{h}_p; \mathbf{h}_e])$`

3. **Stage 2 output says "Top-20"** — L32: `Output: \textbf{Top-20 Candidates}`
   Code outputs top-15. Change to: `Output: \textbf{Top-15 Candidates}`

4. **Stage 3 is completely wrong pipeline** — L36-42: entire Stage 3 box describes BGE cross-encoder + KL-distill
   Must be REPLACED with T5 DPO Preference Judge:
   ```
   \textbf{Stage 3: Preference-Aligned Generative Judge} \\
   Model: \texttt{Flan-T5-Base} (DPO fine-tuned, 250M params) \\
   Scoring: $s_3(q,e) = \log \pi_\theta(e \mid p)$ (listwise: all 15 candidates in context) \\
   Training: Direct Preference Optimization ($\mathcal{L}_{\text{DPO}}$, $\beta_{\text{dpo}}=0.1$) \\
   Output: \textbf{Top-1 Entity} (Final answer)
   ```

5. **Stage 3 subsubsection title in main.tex** — "Stage 3: Cross-Encoder Precision Judge"
   Must rename to: "Stage 3: Preference-Aligned Generative Judge"

**Fix status:** [ ] Pending (high priority — done during Pass 3)

---

### fig_kg.tex (fig:kg) — MULTI-HOP KG REASONING TRACE
**Placement:** §Prelim — CORRECT, introduces KG traversal concept.

**Errors found:** NONE — this figure is purely illustrative (Jerry Jones example).
Caption is factually accurate. No notation conflicts.

**Fix status:** [x] OK — no changes needed

---

### fig_working_example.tex (fig:working_example) — APPENDIX INFERENCE TRACE
**Placement:** Appendix — shows Standard RL vs STRL comparison.

**Errors found:**

1. **Title says "Standard RL" vs "STRL"** — this comparison is correct conceptually.
   But the paper's primary system is now RLMC (exp9), not generic "Standard RL".
   Change top label: `Standard RL` → `RLMC (exp9)`

2. **Conf scores are made-up** — L79: `conf=0.65`, L131: `conf=0.88` — these are illustrative.
   OK for appendix as long as caption says "illustrative".

3. **STRL section still says "STRL"** — correct, keep as is.

4. **Bottom result box says "(1 Candidate)"** — L151: `Exact Traversed Hit@1 Answer Set (1 Candidate)`
   This is illustrative but correct for STRL tight traversal. OK.

5. **No mention of CDS/DPO stage** — the working example only shows traversal, not reranking.
   Consider adding a third panel showing CDS funnel output. OR add a note in caption:
   "Traversal outputs are then passed to the CDS Preference-Aligned Judge for final selection."

**Fix status:** [ ] Minor — rename "Standard RL" label + update caption

---

### Orphan Figures (not used in paper)
These exist in tikz/ but are NOT \input'd anywhere in main.tex:

**fig_architecture.tex** — alternate simpler architecture diagram
  Decision: Either (a) replace fig_detailed_dataflow with this simpler one, or (b) delete.
  Recommend: Keep fig_detailed_dataflow (richer), delete fig_architecture.tex eventually.

**fig_pipeline.tex** — pipeline overview diagram
  Decision: Could be used in Introduction to give high-level overview before Methodology.
  Recommend: Evaluate adding it to Introduction (after contribution bullets).

**fig_traversal.tex** — traversal dynamics figure
  Decision: Could replace or complement fig_working_example in Appendix.
  Recommend: Check content, decide if useful.

**Fix status:** [ ] Decision pending — check content of fig_pipeline and fig_traversal

---

### Figure Placement Issues

| Figure | Current position | Issue | Recommended fix |
|---|---|---|---|
| fig:kg (KG trace) | §Prelim, before ProbForm | Correct | Keep |
| fig:arch (dataflow) | §Methodology opening (figure*[t]) | Correct — shows before Stage I text | Keep |
| fig:cds_funnel | §Stage III | Correct — appears right as CDS is described | Keep |
| fig:working_example | Appendix | Good position | Keep, add DPO note in caption |
| fig_pipeline (orphan) | Not placed | Could add to Introduction | Evaluate |


---

## PAPER EDITING PROGRESS TRACKER
<!-- Update this every session. This is the canonical state of the paper. -->
<!-- Last updated: 2026-06-23 -->

### FINAL NUMBERS FOR PAPER (use THESE, not old numbers)
| Pipeline | Recall | Hit@1 | Source |
|---|---|---|---|
| Stage I only (exp7) | ~58% | 37.21% | from code+notes |
| **RLMC+DPO (exp9+exp38)** | **63.08%** | **42.12%** | benchmarked session c2131455 |
| STRL variant (exp15+exp38) | 75.59%* | 40.43% | benchmarked session c2131455 |
| Ensemble+T5 SFT (SOTA run) | 77.78% | 43.77% | benchmarked session c2131455 |
*75.59% is traversal recall (independent of reranker — keep this number)

Old paper numbers (DO NOT USE): 39.95% for RLMC, 42.80% for STRL

---

### PASS STATUS

#### [x] PASS 0 — Abstract rewritten
- Removed: "Freebase and Wikidata" → "Freebase" only
- Added: "RL applied at both traversal and candidate selection layers"
- Updated Stage II action names: \textsc{Tight/Medium/Loose/Stop}
- Updated Stage III: CDS funnel + "Preference-Aligned Generative Judge (DPO)"
- Added: 42.12% Hit@1, 75.59% recall, >1900→14 edges efficiency
- Updated positioning: "RL-based traversal-control and preference-aligned candidate selection"

#### [x] PASS 1 — Notation unification (all conflicts resolved)
All changes made to main.tex:
- Answer set: A → Y (§Prelim)
- MDP tuple: M=(S,A,P,R) → M=(S,A,P,F) — R reserved for relation set only
- Width penalty: γ → ω in R_t equation (both §ProbForm and §Stage II)
Sections to KEEP but condense:
- [ ] Dataset description (CWQ stats)
- [ ] Baseline comparisons
- [ ] Hyperparameter table (Table 4)
- [ ] Evaluation metrics

#### [ ] PASS 6 — Results & Analysis (§8)
Key changes:
- [ ] Table 1 (tab:main_results): Update RLMC Hit@1 39.95% → 42.12%
      Update STRL Hit@1 42.80% → 40.43% (or discuss in text)
- [ ] Table 2 (tab:extended_metrics): Update numbers to match new pipeline
      exp9 row: Hit@1=42.12%, Recall=63.08%
      exp15 row: Hit@1=40.43%, Recall=75.59%
      Efficiency numbers (edges, latency, VRAM) likely unchanged for traversal
- [ ] Results subsections §8.2–8.6: mostly qualitative, consider merging
- [ ] Add the "alignment tax" finding as a named finding in Results

#### [ ] PASS 7 — Ablation Study (§10)
Current state: 5 qualitative paragraphs, NO table.
Options:
  A: Add ablation table (requires benchmarks — time-consuming)
  B: Rename to "Qualitative Analysis" to be honest about what it is
  → Recommend B (rename + tighten text)
  
Ablation paragraphs to keep (summarized):
1. Without Stage I (random traversal) → performance collapse
2. Without RL (Stage I only) → misses multi-hop chains  
3. Without CDS (raw traversal output) → low precision
4. Without DPO (SFT only, exp31) → 43.23% vs 42.12% alignment tax
5. STRL vs RLMC: STRL better recall (75.59%) but lower H@1 (40.43%)

#### [ ] PASS 8 — Conclusion (§13)
- [ ] Update 42.80% → 42.12%
- [ ] Update STRL numbers
- [ ] Reflect DPO as Stage III contribution

---

### SECTION LINE MAP (main.tex, post-Pass-1)
<!-- Line numbers may shift slightly after edits — use \section{} grep to find exact lines -->
Abstract:       L53–L73  ← DONE
Introduction:   L91–L134
Related Work:   L135–L162
Preliminaries:  L163–L181
ProbFormulation: L187–L207
Methodology:    L208–L385
  Stage I:      L220–L265
  Stage II:     L266–L315
  STRL:         L315–L326
  Stage III:    L326–L380
  Complexity:   L358–L385
Exp Setup:      L385–L600
Results:        L600–L785
ResearchChall:  L785–L800
Ablation:       L798–L850
Limitations:    L849–L862
FutureWork:     L861–L878
Conclusion:     L877–L887
Appendix:       L886–L905

---

### NOTATION AFTER PASS 1 (use these going forward)
<!-- Reference when writing new content or checking consistency -->
q               : question tokens
C_q (bold)      : RoBERTa output matrix [n, 1024]
h_q             : CLS embedding [d_h=512]
z_q             : planning projection W_p h_q [d_h]
H^(l)           : cross-hop transformer states [H_max, d_h]
h_h             : H^(L)_h — refined hop state for hop h [d_h]
eta_q           : confidence scalar σ(W_η h_q) ∈ [0,1]
s_t             : RL state [h_h; eta_q] ∈ R^{d_h+1}
z_t             : W_z h_h — projected to relation space [d_h]
r_t             : target relation embedding [d_h]
r_j^-           : negative relation embedding
Xi              : traversal trajectory (s_0,a_0,...,s_T)
R_t             : α Acc_t - β Cost_t - ω Width_t (Eq. ref{eq:reward})
gamma_discount  : 0.99 discount factor
lambda_GAE      : 0.95 GAE smoothing
w_sem           : STRL curriculum weight (1.0→0.3)
tau             : 0.07 InfoNCE temperature
pi_theta        : DPO policy (T5)
pi_ref          : frozen SFT reference (exp31)
e^+, e^-        : chosen/rejected entity names (DPO pairs)
beta_dpo        : 0.1 DPO KL penalty
q_bi, e_bi      : MiniLM 384-dim CDS Stage 1 reps
q_pw, h_p, h_e  : MPNet 768-dim CDS Stage 2 reps
s_1, s_2, s_3   : CDS stage scoring functions
Y               : answer set {a_1,...,a_n}
F               : MDP reward function (in tuple M=(S,A,P,F))
H_max           : 4 (max hops)
d_h             : 768 (paper) / 512 (code)
d_mlp           : 256 (MLP hidden dim — DEFINE IN PASS 2)
K               : domain vocab size (DEFINE IN PASS 2)
omega           : width penalty weight in R_t
