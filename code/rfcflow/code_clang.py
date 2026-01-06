import os
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Dict, Any


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
    parameters: List[Dict[str, str]] = field(default_factory=list)
    condition: str = ""
    dependencies: Dict[str, Any] = field(default_factory=dict)


def _iter_source_files(root: str, max_files: int) -> Iterator[str]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # deterministic traversal
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            if not (fn.endswith(".c") or fn.endswith(".h") or fn.endswith(".cc") or fn.endswith(".cpp")):
                continue
            yield os.path.join(dirpath, fn)
            count += 1
            if count >= max_files:
                return


def extract_functions_clang(
    code_root: str,
    max_files: int = 2000,
    max_functions: int = 5000,
    libclang_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
) -> List[FunctionRecord]:
    """
    Function extractor based on clang.cindex (preferred for paper reproduction).

    Notes:
    - This is intentionally *lighter* than the original full dependency analyzer: stage-1 relevance only needs function bodies.
    - libclang path and include dirs are configurable so it's runnable outside your local env.
    """
    try:
        import clang.cindex  # type: ignore
        from clang.cindex import Index, CursorKind, TranslationUnit  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "clang python bindings are not installed. Install `clang` package or use --code-backend regex."
        ) from e

    if libclang_path:
        clang.cindex.Config.set_library_file(libclang_path)

    idx = Index.create()
    args: List[str] = []
    for d in include_dirs or []:
        args.append(f"-I{d}")
    if extra_args:
        args.extend(extra_args)

    funcs: List[FunctionRecord] = []
    fid = 1

    for path in _iter_source_files(code_root, max_files=max_files):
        try:
            tu = idx.parse(
                path,
                args=args,
                options=TranslationUnit.PARSE_INCOMPLETE | TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except Exception:
            continue

        # cache file content for slicing
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            content = ""

        for c in tu.cursor.get_children():
            if c.kind != CursorKind.FUNCTION_DECL:
                continue
            if not c.is_definition():
                continue
            if not c.extent.start.file:
                continue
            file_path = str(c.extent.start.file)
            # only keep functions in code_root
            if not os.path.abspath(file_path).startswith(os.path.abspath(code_root)):
                continue

            start_line = c.extent.start.line
            end_line = c.extent.end.line
            name = c.spelling

            # slice by offsets if possible, else by line range
            src = ""
            if content:
                try:
                    src = content[c.extent.start.offset : c.extent.end.offset].strip()
                except Exception:
                    pass
                if not src:
                    lines = content.splitlines()
                    if 0 < start_line <= len(lines) and 0 < end_line <= len(lines):
                        src = "\n".join(lines[start_line - 1 : end_line]).strip()

            funcs.append(
                FunctionRecord(
                    function_id=fid,
                    function_name=name,
                    file_path=os.path.relpath(file_path, code_root),
                    start_line=start_line,
                    end_line=end_line,
                    source_code=src,
                )
            )
            fid += 1
            if len(funcs) >= max_functions:
                return funcs

    return funcs


