import json
import os
from typing import List


def load_protocol_keywords(protocol: str, assets_dir: str) -> List[str]:
    """
    Return a flat list of keywords for the protocol.
    - http: uses assets/keywords/http.json (status codes, header fields, methods, etc.)
    - tls: uses assets/keywords/tls.json (categories with "keywords" list)
    """
    protocol = protocol.lower()
    if protocol == "http":
        path = os.path.join(assets_dir, "keywords", "http.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)["keywords"]
        kws: List[str] = []
        # status codes: include both numbers and descriptions
        for code, desc in data.get("status_code", {}).items():
            kws.append(str(code))
            kws.append(str(desc))
        for k in data.get("field_name", []):
            kws.append(str(k))
        for k in data.get("content_coding", []):
            kws.append(str(k))
        for k in data.get("method", []):
            kws.append(str(k))
        for k in data.get("syntax", []):
            kws.append(str(k))
        # de-dup
        return sorted(set(kws), key=lambda x: (len(x), x))
    elif protocol == "tls":
        path = os.path.join(assets_dir, "keywords", "tls.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kws: List[str] = []
        for _, obj in data.items():
            for k in obj.get("keywords", []):
                kws.append(str(k))
        return sorted(set(kws), key=lambda x: (len(x), x))
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


def match_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Case-insensitive substring match. Returns matched keywords.
    """
    t = text.lower()
    out: List[str] = []
    for kw in keywords:
        if kw and kw.lower() in t:
            out.append(kw)
    return out




