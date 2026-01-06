import re
from typing import List, Dict, Any, Tuple

from code_extract import FunctionRecord
from rfc import SRRecord


def _tokenize(text: str) -> List[str]:
    # keep alnum/_ tokens, uppercase for robust matching
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)
    return [t.upper() for t in toks]


def _score_overlap(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = sa.intersection(sb)
    return len(inter) / max(1, len(sa))


def build_candidates_heuristic(functions: List[FunctionRecord], srs: List[SRRecord], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Paper-friendly baseline:
    - token overlap between function body and SR sentence
    - output top_k SR ids per function
    """
    sr_tokens = [(sr.sr_id, _tokenize(sr.sr)) for sr in srs]
    out: List[Dict[str, Any]] = []

    for fn in functions:
        ftoks = _tokenize(fn.source_code)
        scored: List[Tuple[int, float]] = []
        for sr_id, stoks in sr_tokens:
            sc = _score_overlap(stoks, ftoks)
            if sc > 0:
                scored.append((sr_id, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [sr_id for sr_id, _ in scored[:top_k]]
        out.append(
            {
                "function_id": fn.function_id,
                "function_name": fn.function_name,
                "file_path": fn.file_path,
                "sr_ids": top,
            }
        )
    return out


