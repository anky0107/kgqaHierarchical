"""
utils.py — Path and relation utilities for the CDS pipeline.

All path/relation helper functions live here so every other module
imports from one place.
"""
from __future__ import annotations
from typing import Union


# ─────────────────────────────────────────────────────────────
#  Path serialisation
# ─────────────────────────────────────────────────────────────

def flatten_path(path: Union[str, list, None]) -> str:
    """
    Convert the raw CDS JSON 'path' field to a flat, usable string.

    The field is always stored as list[list[str]] — multi-hop beam
    alternatives per hop, e.g.:
        [
          ['film.film.directed_by', 'film.film.produced_by'],   ← hop 0
          ['people.person.nationality'],                         ← hop 1
        ]

    Strategy: take the FIRST (highest-confidence) relation from each
    hop's alternative list and join with ' → '.

    Handles all edge-cases gracefully so callers never receive an error.
    """
    if path is None:
        return ""
    if isinstance(path, str):
        return path.strip()
    if isinstance(path, list):
        if not path:
            return ""
        # list[list[str]] — standard CDS format
        if isinstance(path[0], list):
            hops = [inner[0] for inner in path if inner]
        else:
            # list[str] — already flat
            hops = [str(r) for r in path]
        return " -> ".join(hops)
    return str(path)


# ─────────────────────────────────────────────────────────────
#  Relation ID → natural language
# ─────────────────────────────────────────────────────────────

def rel_to_nl(rel_id: str) -> str:
    """
    Convert a Freebase relation ID to a short natural-language phrase.
    Uses the same heuristic as RelationEmbeddingBank._rel_to_text in
    exp15_strl.py for consistency across the codebase.

    Examples
    --------
    "film.film.directed_by"       -> "film directed by"
    "people.person.nationality"   -> "person nationality"
    "award.award_winner.awards_won" -> "award winner HAS awards won"
    """
    parts = rel_id.split(".")
    if len(parts) >= 3:
        subject   = parts[-2].replace("_", " ")
        predicate = parts[-1].replace("_", " ")
        if (predicate.endswith("s")
                or "owned"   in predicate
                or "founded" in predicate
                or "won"     in predicate):
            return f"{subject} HAS {predicate}"
        return f"{subject} {predicate}"
    if len(parts) == 2:
        return parts[-1].replace("_", " ")
    return rel_id.replace(".", " ").replace("_", " ")


def path_to_nl(path: Union[str, list, None]) -> str:
    """
    Flatten 'path' and convert each relation ID to natural language.

    "film.film.directed_by -> people.person.nationality"
      → "film directed by -> person nationality"
    """
    flat = flatten_path(path)
    if not flat:
        return ""
    hops = [h.strip() for h in flat.split(" -> ") if h.strip()]
    return " -> ".join(rel_to_nl(h) for h in hops)
