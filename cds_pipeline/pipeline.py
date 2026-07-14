"""
pipeline.py — CDSPipeline: the clean 3-stage cascade.

Bugs fixed vs original benchmark scripts
-----------------------------------------
1. Stage 2 now calls PathAwareRanker.forward() (MLP fusion head).
   Original scripts loaded only the MPNet encoder and used
   F.cosine_similarity(q_emb + p_emb, e_emb) — completely ignoring
   the trained fusion weights.

2. Path field correctly flattened from list[list[str]] to a string
   before passing to tokenizers.

3. Stage 2 uses the MPNet tokenizer throughout. The original
   benchmark_final_cds_v3.py accidentally used the S1 (MiniLM)
   tokenizer for S2 inputs.

4. Adaptive S1 top-k: keeps up to 200 candidates for large beams
   (previously hard-capped at 100, causing gold-entity loss for
   questions with >100 candidates).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Union

from .models import load_stage1, load_stage2, load_stage3
from .utils  import flatten_path


class CDSPipeline:
    """
    Cascading Dust Separator — three-stage candidate ranker.

    Parameters
    ----------
    device      : torch device (default: CUDA if available, else CPU)
    s3_version  : 'v2' (name-only) or 'v3' (path-aware Stage 3)
    s1_top_k    : max candidates kept after Stage 1 (default 200)
    s2_top_k    : max candidates kept after Stage 2 (default 15)
    """

    def __init__(
        self,
        device:      Optional[torch.device] = None,
        s3_version:  str = "v2",
        s2_version:  str = "v1",
        s1_top_k:    int = 200,
        s2_top_k:    int = 50,
        bypass_stage1: bool = False,
        bypass_stage2: bool = False,
    ) -> None:
        self.device     = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.s3_version = s3_version
        self.s2_version = s2_version
        self.s1_top_k   = s1_top_k
        self.s2_top_k   = s2_top_k
        self.bypass_stage1 = bypass_stage1
        self.bypass_stage2 = bypass_stage2

        print(f"[CDS] Device={self.device}  S2={s2_version}  S3={s3_version}  "
              f"bypass_s1={bypass_stage1}  S2-top={s2_top_k}")
        self.s1_tok, self.s1_model = load_stage1(self.device)
        self.s2_tok, self.s2_model = load_stage2(self.device, version=s2_version)
        self.s3_tok, self.s3_model = load_stage3(self.device, version=s3_version)
        print("[CDS] Ready.")

    # ── public API ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def rank(
        self,
        question:   str,
        candidates: List[Dict[str, Any]],
        path:       Union[str, list, None] = None,
        return_intermediates: bool = False,
    ) -> Union[List[Dict[str, Any]], tuple]:
        """
        Score and sort candidates for one question.

        Parameters
        ----------
        question   : natural-language question string
        candidates : list of dicts, each with at least a 'name' key
        path       : raw 'path' value from the CDS JSON
                     (str, list[str], or list[list[str]] — all handled)
        bypass_s1  : if True, skips Stage 1 (Bi-Encoder)
        bypass_s2  : if True, skips Stage 2 (MPNet Path-Aware)
        return_intermediates: if True, returns (final_ranked, cands_after_s1, cands_after_s2)

        Returns
        -------
        Candidates sorted by descending relevance (best first).
        """
        if not candidates:
            return ([], [], []) if return_intermediates else []

        global_path_str = flatten_path(path)   # ← BUG FIX: handles list[list[str]]

        # ── Stage 1: Bi-Encoder ─────────────────────────────────────────────
        if self.bypass_stage1:
            cands_s2 = candidates
        else:
            cands_s2 = self._stage1(question, candidates)

        # ── Stage 2: PathAwareRanker (MLP fusion) ───────────────────────────
        if self.bypass_stage2:
            cands_s3 = cands_s2
        else:
            cands_s3 = self._stage2(question, global_path_str, cands_s2)

        # ── Stage 3: Cross-Encoder ──────────────────────────────────────────
        final_ranked = self._stage3(question, global_path_str, cands_s3)
        
        if return_intermediates:
            return final_ranked, cands_s2, cands_s3
        return final_ranked

    # ── stage helpers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _stage1(
        self,
        question:   str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bi-encoder cosine-similarity pruning → top s1_top_k."""
        names = [c.get("name", "") for c in candidates]

        qe  = self.s1_tok(
            question, return_tensors="pt",
            padding=True, truncation=True, max_length=128,
        ).to(self.device)
        qv  = self.s1_model(**qe).last_hidden_state[:, 0, :]   # [1, 384]

        # Chunk entity encoding to avoid OOM on very large beams
        chunks = []
        for i in range(0, len(names), 512):
            chunk = self.s1_tok(
                names[i : i + 512], return_tensors="pt",
                padding=True, truncation=True, max_length=64,
            ).to(self.device)
            chunks.append(self.s1_model(**chunk).last_hidden_state[:, 0, :])
        ev = torch.cat(chunks, dim=0)                           # [N, 384]

        scores  = F.cosine_similarity(qv, ev)
        top_k   = min(self.s1_top_k, len(candidates))
        top_idx = torch.topk(scores, top_k).indices.cpu().tolist()
        return [candidates[i] for i in top_idx]

    @torch.no_grad()
    def _stage2(
        self,
        question:    str,
        path_str:    str,
        candidates:  List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        PathAwareRanker (MPNet + MLP fusion) → top s2_top_k.

        BUG FIX: calls model.forward(q_ids, q_mask, p_ids, p_mask, e_ids, e_mask)
        instead of F.cosine_similarity(q_emb + p_emb, e_emb).

        Exp 21: Uses per-candidate 'path' field when present (beam-search
        produced paths), falling back to the global path_str otherwise.
        """
        names        = [c.get("name", "") for c in candidates]
        # Per-candidate path strings (Exp 21 beam paths override global path)
        path_strings = [c.get("path") or path_str for c in candidates]
        n            = len(names)

        # ── Chunked forward pass to avoid OOM ─
        # Use autocast and larger chunk size to process 5,000+ candidates quickly
        S2_CHUNK = 64
        all_scores = []
        with torch.amp.autocast('cuda'):
            for i in range(0, n, S2_CHUNK):
                ns  = names[i : i + S2_CHUNK]
                ps  = path_strings[i : i + S2_CHUNK]
                nc  = len(ns)
                qe = self.s2_tok(
                    [question] * nc, padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                ).to(self.device)
                pe = self.s2_tok(
                    ps, padding=True, truncation=True,
                    max_length=64, return_tensors="pt",
                ).to(self.device)
                ee = self.s2_tok(
                    ns, padding=True, truncation=True,
                    max_length=64, return_tensors="pt",
                ).to(self.device)
                chunk_scores = self.s2_model(
                    qe["input_ids"], qe["attention_mask"],
                    pe["input_ids"], pe["attention_mask"],
                    ee["input_ids"], ee["attention_mask"],
                )
                all_scores.append(chunk_scores)

        scores  = torch.cat(all_scores, dim=0)
        top_k   = min(self.s2_top_k, len(candidates))
        top_idx = torch.topk(scores, top_k).indices.cpu().tolist()
        return [candidates[i] for i in top_idx]

    @torch.no_grad()
    def _stage3(
        self,
        question:   str,
        path_str:   str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        BGE cross-encoder final reranking.

        Exp 21: For v4/v5 models, uses the per-candidate 'path' field when
        present so each candidate is scored against its own beam path rather
        than a single global path string.
        """
        names = [c.get("name", "") for c in candidates]

        if self.s3_version in ["v8_gen", "v12_t5_mc", "v18_t5_dpo"]:
            from .utils import path_to_nl
            prompt = f"Question: {question}\n\nCandidates:\n"
            for i, c in enumerate(candidates, 1):
                name = c.get("name", "").strip() or "[UNK]"
                cand_path_str = c.get("path") or path_str or ""
                path_nl = path_to_nl(cand_path_str)
                if path_nl:
                    prompt += f"{i}. {name} (Path: {path_nl})\n"
                else:
                    prompt += f"{i}. {name}\n"
            prompt += "\nWhich of the above candidates is the correct answer to the question? Answer with the exact name."
            
            enc = self.s3_tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            out_ids = self.s3_model.generate(**enc, max_length=64, num_beams=4, early_stopping=True)
            pred_name = self.s3_tok.decode(out_ids[0], skip_special_tokens=True).strip().lower()
            
            # Find the generated name in candidates
            for c in candidates:
                if c.get("name", "").strip().lower() == pred_name:
                    c["score"] = 999.0  # Force to top
                else:
                    c["score"] = 0.0
            
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates

        if self.s3_version in ["v15_t5_listwise", "v16_bge_cross", "v17_bge_infonce"]:
            from .utils import path_to_nl
            # Build one input string per candidate
            if self.s3_version == "v15_t5_listwise":
                inputs = []
                for c in candidates:
                    name = c.get("name", "").strip() or "[UNK]"
                    cand_path_str = c.get("path") or path_str or ""
                    path_nl = path_to_nl(cand_path_str)
                    text = f"{question} | {name} | {path_nl}" if path_nl else f"{question} | {name}"
                    inputs.append(text)
                
                # Score all candidates in chunks
                CHUNK = 32
                all_scores = []
                for i in range(0, len(inputs), CHUNK):
                    enc = self.s3_tok(
                        inputs[i : i + CHUNK],
                        padding=True, truncation=True,
                        max_length=128, return_tensors="pt"
                    ).to(self.device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        chunk_scores = self.s3_model(
                            enc["input_ids"], enc["attention_mask"]
                        )
                    all_scores.append(chunk_scores)
    
                scores = torch.cat(all_scores, dim=0).cpu()
                
            else: # v16_bge_cross
                questions = []
                cand_texts = []
                for c in candidates:
                    name = c.get("name", "").strip() or "[UNK]"
                    cand_path_str = c.get("path") or path_str or ""
                    path_nl = path_to_nl(cand_path_str)
                    cand_text = f"{name} | {path_nl}" if path_nl else name
                    questions.append(question)
                    cand_texts.append(cand_text)
                    
                CHUNK = 32
                all_scores = []
                for i in range(0, len(questions), CHUNK):
                    enc = self.s3_tok(
                        questions[i : i + CHUNK],
                        cand_texts[i : i + CHUNK],
                        padding=True, truncation=True,
                        max_length=256, return_tensors="pt"
                    ).to(self.device)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.s3_model(**enc)
                        chunk_scores = outputs.logits.squeeze(-1)
                    all_scores.append(chunk_scores)
                    
                scores = torch.cat(all_scores, dim=0).cpu()

            ranked = sorted(
                zip(candidates, scores.tolist()),
                key=lambda x: x[1], reverse=True
            )
            return [c for c, _ in ranked]

        elif self.s3_version == "v14_t5_pointer":
            from .utils import path_to_nl
            import re
            prompt = f"Question: {question}\n\nCandidates:\n"
            for i, c in enumerate(candidates, 1):
                name = c.get("name", "").strip() or "[UNK]"
                cand_path_str = c.get("path") or path_str or ""
                path_nl = path_to_nl(cand_path_str)
                if path_nl:
                    prompt += f"{i}. {name} (Path: {path_nl})\n"
                else:
                    prompt += f"{i}. {name}\n"
            prompt += "\nWhich of the above candidates is the correct answer to the question? Output the Candidate Index and reason through the relations."
            
            enc = self.s3_tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            out_ids = self.s3_model.generate(**enc, max_length=128, num_beams=4, early_stopping=True)
            pred_text = self.s3_tok.decode(out_ids[0], skip_special_tokens=True).strip()
            
            # Parse Pointer output
            match = re.search(r'Candidate Index:\s*\[(\d+)\]', pred_text, re.IGNORECASE)
            
            for c in candidates:
                c["score"] = 0.0
                
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= len(candidates):
                    candidates[idx - 1]["score"] = 999.0  # Force to top
            
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates

        elif self.s3_version == "v13_t5_cot":
            from .utils import path_to_nl
            import re
            prompt = f"Question: {question}\n\nCandidates:\n"
            for i, c in enumerate(candidates, 1):
                name = c.get("name", "").strip() or "[UNK]"
                cand_path_str = c.get("path") or path_str or ""
                path_nl = path_to_nl(cand_path_str)
                if path_nl:
                    prompt += f"{i}. {name} (Path: {path_nl})\n"
                else:
                    prompt += f"{i}. {name}\n"
            prompt += "\nWhich of the above candidates is the correct answer to the question? Reason through the relations and output the answer."
            
            enc = self.s3_tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            out_ids = self.s3_model.generate(**enc, max_length=128, num_beams=4, early_stopping=True)
            pred_text = self.s3_tok.decode(out_ids[0], skip_special_tokens=True).strip()
            
            # Parse CoT output
            match = re.search(r'Answer:\s*(.*)', pred_text, re.IGNORECASE)
            if match:
                pred_name = match.group(1).strip().lower()
            else:
                lines = pred_text.split('\n')
                pred_name = lines[-1].strip().lower()
            
            # Find the generated name in candidates
            for c in candidates:
                if c.get("name", "").strip().lower() == pred_name:
                    c["score"] = 999.0  # Force to top
                else:
                    c["score"] = 0.0
            
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates

        elif self.s3_version == "v11_gen_sc":
            from .utils import path_to_nl
            import re
            from collections import Counter
            prompt = f"Question: {question}\n\nCandidates:\n"
            for i, c in enumerate(candidates, 1):
                name = c.get("name", "").strip() or "[UNK]"
                cand_path_str = c.get("path") or path_str or ""
                path_nl = path_to_nl(cand_path_str)
                if path_nl:
                    prompt += f"{i}. {name} (Path: {path_nl})\n"
                else:
                    prompt += f"{i}. {name}\n"
            prompt += "\nExplain your reasoning step-by-step. Then output the final exact name of the correct candidate on a new line prefixed with 'Answer: '."
            
            enc = self.s3_tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            out_ids = self.s3_model.generate(
                **enc, max_length=128, num_return_sequences=10, 
                do_sample=True, temperature=0.7, top_p=0.9, early_stopping=True
            )
            
            predictions = []
            for out_id in out_ids:
                pred_text = self.s3_tok.decode(out_id, skip_special_tokens=True).strip()
                # Parse CoT output
                match = re.search(r'Answer:\s*(.*)', pred_text, re.IGNORECASE)
                if match:
                    predictions.append(match.group(1).strip().lower())
                else:
                    # Fallback to the last line if 'Answer:' isn't found
                    lines = pred_text.split('\n')
                    predictions.append(lines[-1].strip().lower())
            
            # Majority vote
            if predictions:
                pred_name = Counter(predictions).most_common(1)[0][0]
            else:
                pred_name = ""

            # Find the generated name in candidates
            for c in candidates:
                if c.get("name", "").strip().lower() == pred_name:
                    c["score"] = 999.0  # Force to top
                else:
                    c["score"] = 0.0
            
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates

        if self.s3_version == "v3":
            query = f"{question} [PATH] {path_str}"
            texts = names
        elif self.s3_version in ["v7", "v9_rl_policy"]:
            # v7 and v9_rl_policy: name | path_nl  (clean path-only enrichment, no entity type)
            query = question
            from .utils import path_to_nl
            texts = []
            for c in candidates:
                cand_path_str = c.get("path") or path_str
                path_nl  = path_to_nl(cand_path_str)
                parts = [c.get("name", "").strip() or "[UNK]"]
                if path_nl.strip():
                    parts.append(path_nl.strip())
                texts.append(" | ".join(parts))
        elif self.s3_version == "v10_pure_rl":
            from .utils import path_to_nl
            from .rl_features import extract_features
            feature_matrix = []
            for c in candidates:
                cand_path_str = c.get("path") or path_str
                path_nl = path_to_nl(cand_path_str)
                name = c.get("name", "")
                feat = extract_features(question, name, path_nl)
                feature_matrix.append(feat)
            
            feat_tensor = torch.tensor(feature_matrix, dtype=torch.float32, device=self.device)
            logits = self.s3_model(feat_tensor).cpu()
            
            for c, logit in zip(candidates, logits.tolist()):
                c["score"] = logit
            
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates
        elif self.s3_version in ["v4", "v5", "v6"]:
            query = question
            from .utils import path_to_nl
            texts = []
            for c in candidates:
                # Use per-candidate path (beam search) if available
                cand_path_str = c.get("path") or path_str
                path_nl  = path_to_nl(cand_path_str)
                ent_type = c.get("type") or c.get("entity_type") or ""
                parts = [c.get("name", "").strip() or "[UNK]"]
                if path_nl.strip():  parts.append(path_nl.strip())
                if ent_type.strip(): parts.append(ent_type.strip())
                texts.append(" | ".join(parts))
        else:
            query = question
            texts = names

        enc = self.s3_tok(
            [query] * len(texts), texts,
            padding=True, truncation=True,
            max_length=192, return_tensors="pt",
        ).to(self.device)

        logits = self.s3_model(**enc).logits.squeeze(-1).cpu()
        ranked = sorted(
            zip(candidates, logits.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return [c for c, _ in ranked]
