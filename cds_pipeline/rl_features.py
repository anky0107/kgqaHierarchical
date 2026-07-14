import re
from difflib import SequenceMatcher

def tokenize(text: str) -> set:
    """Extract lowercase word tokens from a string."""
    text = str(text).lower()
    tokens = set(re.findall(r'\b\w+\b', text))
    return tokens

def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets of tokens."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def string_similarity(s1: str, s2: str) -> float:
    """Compute Levenshtein-like similarity ratio between two strings."""
    return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()

def extract_features(question: str, cand_name: str, path_nl: str) -> list[float]:
    """
    Extract a lightweight feature vector for a candidate.
    Features:
    0: Jaccard similarity between Question and Candidate Name
    1: Jaccard similarity between Question and Path
    2: String Similarity Ratio between Question and Candidate Name
    3: Exact Substring Match (1.0 if Cand Name is in Question, else 0.0)
    4: Path Length (number of tokens in path_nl) normalized roughly
    """
    q_tokens = tokenize(question)
    name_tokens = tokenize(cand_name)
    path_tokens = tokenize(path_nl)

    f0 = jaccard_similarity(q_tokens, name_tokens)
    f1 = jaccard_similarity(q_tokens, path_tokens)
    f2 = string_similarity(question, cand_name)
    
    cand_lower = str(cand_name).lower()
    q_lower = str(question).lower()
    f3 = 1.0 if (cand_lower and cand_lower in q_lower) else 0.0
    
    # Path length: assume each relation is a few words, normalize by 10
    f4 = min(1.0, len(path_tokens) / 10.0)

    return [f0, f1, f2, f3, f4]
