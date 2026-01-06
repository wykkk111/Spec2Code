import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class SRRecord:
    """
    SR record aligned with the original pipeline intent:
    - sr: the normative sentence containing RFC2119 keywords
    - context_before/after: neighbor PARAGRAPHS (not just neighbor sentences)
    """

    sr_id: int
    sr: str
    keywords: str  # comma separated RFC2119 keywords found
    keywords_list: List[str]
    group: Optional[str] = None
    subgroup: Optional[str] = None
    # Paper-aligned context fields (kept similar to `final/src/parse_rfc.py` outputs)
    section_number: str = ""
    section_title: str = ""
    section_level: int = 0
    para_index: int = 0
    sentence_index: int = 0
    current_para: str = ""
    prev_para: str = ""
    next_para: str = ""
    # Back-compat aliases
    context_before: str = ""
    context_after: str = ""


class RFCParserLite:
    """
    Ported from RFC/final/src/parse_rfc.py (same parsing heuristics),
    but returns SR records instead of writing files.
    """

    def __init__(self) -> None:
        self.section_pattern = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$")
        self.sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        self.format_start = re.compile(r"^\s*(?:Preferred format:|[A-Za-z-]+ = |\+={3,}|\|)")
        self.format_marker = "Preferred format:"
        self.table_marker = "+====="
        self.enum_item_pattern = re.compile(r"^\s{3}(?:\d{1,2})\.\s+")

    def split_sentences(self, text: str) -> List[str]:
        parts = self.sentence_pattern.split(text)
        return [p.strip() for p in parts if p.strip()]

    def is_section_header(self, line: str):
        if line.startswith(" "):
            return False, None, None
        m = self.section_pattern.match(line.strip())
        if m:
            return True, m.group(1), m.group(2)
        return False, None, None

    def is_format_para(self, lines: List[str]) -> bool:
        if not lines:
            return False
        first = lines[0].rstrip()
        return bool(self.format_start.match(first)) or ("Preferred format:" in first)

    def is_exact_para_indent(self, line: str) -> bool:
        return bool(re.match(r"^\s{3,}\S", line))

    def is_enum_item(self, line: str) -> bool:
        return bool(self.enum_item_pattern.match(line))

    def parse_paragraphs(self, lines: List[str]) -> List[Dict[str, Any]]:
        paragraphs: List[Dict[str, Any]] = []
        current: List[str] = []
        looking_for_start = True
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()

            if current and self.is_enum_item(line):
                current.append(line)
                i += 1
                continue

            if not line:
                if current:
                    next_idx = i + 1
                    while next_idx < len(lines) and not lines[next_idx].strip():
                        next_idx += 1
                    if next_idx >= len(lines) or not self.is_enum_item(lines[next_idx]):
                        text = "\n".join(current)
                        if not self.is_format_para(current):
                            paragraphs.append({"content": text, "sentences": self.split_sentences(text)})
                        current = []
                        looking_for_start = True
                i += 1
                continue

            if looking_for_start:
                if self.is_exact_para_indent(line):
                    current = [line]
                    looking_for_start = False
            else:
                current.append(line)
            i += 1

        if current:
            text = "\n".join(current)
            if not self.is_format_para(current):
                paragraphs.append({"content": text, "sentences": self.split_sentences(text)})
        return paragraphs

    def parse_document(self, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        root = {"sections": []}
        stack: List[Dict[str, Any]] = []
        current_content: List[str] = []
        level4_content: List[str] = []

        def handle_current():
            if not current_content or not stack:
                return
            if level4_content:
                first_para_end = 0
                empty_found = False
                for j, l in enumerate(current_content):
                    if not l.strip():
                        if empty_found:
                            first_para_end = j
                            break
                        empty_found = True
                if first_para_end == 0:
                    first_para_end = len(current_content)
                merged = level4_content + current_content[:first_para_end]
                rest = current_content[first_para_end:]
                all_paras: List[Dict[str, Any]] = []
                if merged:
                    all_paras.extend(self.parse_paragraphs(merged))
                if rest:
                    all_paras.extend(self.parse_paragraphs(rest))
                stack[-1]["paragraphs"] = all_paras
            else:
                stack[-1]["paragraphs"] = self.parse_paragraphs(current_content)

        for line in lines:
            is_sec, num, title = self.is_section_header(line)
            if is_sec:
                level = len(num.split("."))
                if level == 4:
                    level4_content = [line]
                    continue
                handle_current()
                level4_content = []

                new_sec = {
                    "number": num,
                    "title": title,
                    "level": min(level, 3),
                    "paragraphs": [],
                    "sections": [],
                }
                if not stack:
                    root["sections"].append(new_sec)
                    stack = [new_sec]
                else:
                    while stack and stack[-1]["level"] >= min(level, 3):
                        stack.pop()
                    if not stack:
                        root["sections"].append(new_sec)
                    else:
                        stack[-1]["sections"].append(new_sec)
                    stack.append(new_sec)
                current_content = []
            else:
                current_content.append(line)

        handle_current()
        return root

    @staticmethod
    def clean_sentence(content: str) -> str:
        content = content.replace("\n   ", " ")
        content = content.replace("\n", " ")
        content = " ".join(content.split())
        content = content.replace("- ", "-")
        return content


RFC2119_PATTERNS = [
    ("MUST NOT", r"\bMUST\s+NOT\b"),
    ("SHALL NOT", r"\bSHALL\s+NOT\b"),
    ("SHOULD NOT", r"\bSHOULD\s+NOT\b"),
    ("NOT RECOMMENDED", r"\bNOT\s+RECOMMENDED\b"),
    ("MUST", r"\bMUST\b(?!\s+NOT)"),
    ("REQUIRED", r"\bREQUIRED\b"),
    ("SHALL", r"\bSHALL\b(?!\s+NOT)"),
    ("SHOULD", r"\bSHOULD\b(?!\s+NOT)"),
    ("RECOMMENDED", r"(?<!NOT )\bRECOMMENDED\b"),
    ("MAY", r"\bMAY\b"),
    ("OPTIONAL", r"\bOPTIONAL\b"),
]


def extract_srs_from_rfc_text(rfc_text: str, rfc_id: str = "", max_srs: int = 2000) -> List[SRRecord]:
    """
    Paper-compatible SR extraction:
    - parse sections/paragraphs
    - re-split sentences from paragraph content
    - match RFC2119 keywords with the same priority rules as original
    - attach prev/next PARAGRAPH as context
    """
    parser = RFCParserLite()
    doc = parser.parse_document(rfc_text)

    out: List[SRRecord] = []
    sr_id = 0

    def walk(sec: Dict[str, Any]) -> None:
        nonlocal sr_id
        section_number = str(sec.get("number", ""))
        section_title = str(sec.get("title", ""))
        section_level = int(sec.get("level", 0) or 0)

        paras = [p.get("content", "") for p in sec.get("paragraphs", [])]
        # Use (section_number, para_index) adjacency (paper behavior) rather than cross-section flatten.
        for para_index, para_content in enumerate(paras, 1):
            prev_para = paras[para_index - 2] if para_index - 2 >= 0 else ""
            next_para = paras[para_index] if para_index < len(paras) else ""

            # Re-split sentences from paragraph content
            for sentence_index, sent in enumerate(parser.split_sentences(para_content), 1):
                found: List[str] = []
                for kw, pat in RFC2119_PATTERNS:
                    if re.search(pat, sent):
                        found.append(kw)
                if not found:
                    continue

                cleaned_sent = parser.clean_sentence(sent)
                cleaned_prev = parser.clean_sentence(prev_para) if prev_para else ""
                cleaned_next = parser.clean_sentence(next_para) if next_para else ""
                cleaned_cur = parser.clean_sentence(para_content) if para_content else ""

                out.append(
                    SRRecord(
                        sr_id=sr_id,
                        sr=cleaned_sent,
                        keywords=",".join(found),
                        keywords_list=found,
                        section_number=section_number,
                        section_title=section_title,
                        section_level=section_level,
                        para_index=para_index,
                        sentence_index=sentence_index,
                        current_para=cleaned_cur,
                        prev_para=cleaned_prev,
                        next_para=cleaned_next,
                        context_before=cleaned_prev,
                        context_after=cleaned_next,
                    )
                )
                sr_id += 1
                if len(out) >= max_srs:
                    return

        for child in sec.get("sections", []) or []:
            if len(out) >= max_srs:
                return
            walk(child)

    for s in doc.get("sections", []):
        if len(out) >= max_srs:
            break
        walk(s)

    return out


