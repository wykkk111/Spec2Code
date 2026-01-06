import os
import re
from dataclasses import dataclass, field
from typing import List, Iterator, Tuple, Optional, Dict, Any


@dataclass
class FunctionRecord:
    function_id: int
    function_name: str
    file_path: str
    start_line: int
    end_line: int
    source_code: str
    # Optional rich metadata (paper-compatible)
    return_type: str = ""
    parameters: List[Dict[str, str]] = field(default_factory=list)  # [{"name":..,"type":..}]
    condition: str = ""
    dependencies: Dict[str, Any] = field(default_factory=dict)


_func_sig = re.compile(
    r"""
    (?P<ret>^[A-Za-z_][\w\s\*\(\)]*?)       # return type-ish
    \s+
    (?P<name>[A-Za-z_]\w*)                 # function name
    \s*\(
        (?P<args>[^;{}]*)
    \)\s*
    \{                                      # body start
    """,
    re.MULTILINE | re.VERBOSE,
)


def _iter_c_files(root: str, max_files: int) -> Iterator[str]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # deterministic traversal
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            if not (fn.endswith(".c") or fn.endswith(".h") or fn.endswith(".cc") or fn.endswith(".cpp")):
                continue
            path = os.path.join(dirpath, fn)
            yield path
            count += 1
            if count >= max_files:
                return


def _extract_brace_block(text: str, start_idx: int) -> Tuple[str, int]:
    """
    Given text and index at '{', return (block_text, end_idx_exclusive).
    Very small brace matcher, ignores strings/comments (acceptable for baseline).
    """
    depth = 0
    i = start_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1], i + 1
        i += 1
    return text[start_idx:], n


def extract_functions_regex(code_root: str, max_files: int = 2000, max_functions: int = 5000) -> List[FunctionRecord]:
    """
    Minimal C/C++ function extractor via regex + brace matching.
    It's not perfect, but it's runnable everywhere without libclang.
    """
    functions: List[FunctionRecord] = []
    fid = 1
    for path in _iter_c_files(code_root, max_files=max_files):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        for m in _func_sig.finditer(text):
            name = m.group("name")
            brace_idx = m.end() - 1  # points to '{'
            body, end_idx = _extract_brace_block(text, brace_idx)

            # line numbers (best-effort)
            start_line = text[: m.start()].count("\n") + 1
            end_line = text[: end_idx].count("\n") + 1

            functions.append(
                FunctionRecord(
                    function_id=fid,
                    function_name=name,
                    file_path=os.path.relpath(path, code_root),
                    start_line=start_line,
                    end_line=end_line,
                    source_code=body,
                )
            )
            fid += 1
            if len(functions) >= max_functions:
                return functions

    return functions


def extract_functions(
    code_root: str,
    backend: str = "regex",
    max_files: int = 2000,
    max_functions: int = 5000,
    libclang_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
) -> List[FunctionRecord]:
    """
    Unified entry point:
    - backend=clang: use clang AST (preferred, paper-compatible)
    - backend=regex: lightweight fallback
    """
    backend = (backend or "regex").lower()
    if backend == "clang":
        from code_clang import extract_functions_clang

        clang_funcs = extract_functions_clang(
            code_root=code_root,
            max_files=max_files,
            max_functions=max_functions,
            libclang_path=libclang_path,
            include_dirs=include_dirs,
            extra_args=extra_args,
        )
        return [
            FunctionRecord(
                function_id=f.function_id,
                function_name=f.function_name,
                file_path=f.file_path,
                start_line=f.start_line,
                end_line=f.end_line,
                source_code=f.source_code,
            )
            for f in clang_funcs
        ]

    if backend != "regex":
        raise ValueError(f"Unknown code backend: {backend}")
    return extract_functions_regex(code_root, max_files=max_files, max_functions=max_functions)


