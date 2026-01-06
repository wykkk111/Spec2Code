import glob
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Iterator, Tuple

import clang.cindex  # type: ignore
from clang.cindex import Index, CursorKind, TranslationUnit, StorageClass  # type: ignore

from code_extract import FunctionRecord


@dataclass
class MacroInfo:
    name: str
    value: str
    location: str
    is_constant: bool = False
    is_function_like: bool = False
    parameters: List[str] = field(default_factory=list)
    definition: str = ""
    condition: str = ""


@dataclass
class PreprocessorCondition:
    condition: str
    start_line: int
    end_line: int = -1
    parent: Optional["PreprocessorCondition"] = None
    children: List["PreprocessorCondition"] = field(default_factory=list)


@dataclass
class GlobalVarInfo:
    name: str
    type_str: str
    location: str
    is_extern: bool
    is_static: bool
    definition: str
    initializer: Optional[str] = None
    full_definition: Optional[str] = None
    used_macros: Dict[str, MacroInfo] = field(default_factory=dict)
    condition: str = ""


@dataclass
class StructInfo:
    name: str
    definition: str
    location: str
    is_complete: bool = False
    typedef_name: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)
    condition: str = ""


@dataclass
class _FuncInternal:
    key: str  # name@location
    id: int
    name: str
    file: str
    location: str
    start_line: int
    end_line: int
    return_type: str
    parameters: List[Dict[str, str]]
    condition: str
    source_code: str
    calls: Dict[str, str] = field(default_factory=dict)  # called_name -> snippet/decl
    callers: Set[str] = field(default_factory=set)  # keys of caller functions
    used_macros: Dict[str, MacroInfo] = field(default_factory=dict)
    used_structs: Dict[str, StructInfo] = field(default_factory=dict)
    used_globals: Dict[str, GlobalVarInfo] = field(default_factory=dict)
    used_typedefs: Set[str] = field(default_factory=set)


class CodeAnalyzer:
    """
    Paper-aligned dependency extractor based on `final/src/code_analyzer.py`,
    but configurable (no hardcoded include paths or libclang path).
    """

    def __init__(
        self,
        code_root: str,
        libclang_path: Optional[str] = None,
        include_dirs: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None,
        max_files: int = 2000,
        max_functions: int = 5000,
    ) -> None:
        self.code_root = os.path.abspath(code_root)
        if libclang_path:
            clang.cindex.Config.set_library_file(libclang_path)

        self.index = Index.create()
        self.max_files = max_files
        self.max_functions = max_functions
        self.file_patterns = file_patterns or ["*.c", "*.h", "*.cc", "*.cpp", "*.cxx", "*.hpp"]

        self.compile_args: List[str] = []
        for d in include_dirs or []:
            self.compile_args.append(f"-I{d}")
        if extra_args:
            self.compile_args.extend(extra_args)

        self.file_contents: Dict[str, str] = {}
        self.processed_files: Set[str] = set()

        self.preprocessor_conditions: Dict[str, List[PreprocessorCondition]] = {}
        self.macro_cache: Dict[str, MacroInfo] = {}
        self.structs: Dict[str, StructInfo] = {}
        self.globals: Dict[str, GlobalVarInfo] = {}
        self.typedef_map: Dict[str, str] = {}

        self.functions: Dict[str, _FuncInternal] = {}
        self._func_id_counter = 1
        self._current_function_key: Optional[str] = None

    def _iter_source_files(self) -> Iterator[str]:
        # Deterministic file order: collect then sort.
        paths: List[str] = []
        for pattern in self.file_patterns:
            full_pattern = os.path.join(self.code_root, "**", pattern)
            for file_path in glob.glob(full_pattern, recursive=True):
                file_path = os.path.abspath(file_path)
                if file_path in self.processed_files:
                    continue
                paths.append(file_path)
        paths = sorted(set(paths))
        for i, p in enumerate(paths):
            if i >= self.max_files:
                return
            yield p

    def _read_file_content(self, file_path: str) -> str:
        if file_path not in self.file_contents:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.file_contents[file_path] = f.read()
            except UnicodeDecodeError:
                with open(file_path, "rb") as f:
                    self.file_contents[file_path] = f.read().decode("latin-1")
            except Exception:
                self.file_contents[file_path] = ""
        return self.file_contents[file_path]

    def _extract_source_code(self, node) -> str:
        if not node.extent.start.file:
            return ""
        file_path = os.path.abspath(str(node.extent.start.file))
        content = self._read_file_content(file_path)
        if not content:
            return ""
        start_offset = node.extent.start.offset
        end_offset = node.extent.end.offset
        extracted = ""
        if 0 <= start_offset < len(content) and 0 <= end_offset <= len(content) and start_offset < end_offset:
            extracted = content[start_offset:end_offset].strip()
        if not extracted:
            # fallback by line span
            lines = content.splitlines()
            sl = node.extent.start.line
            el = node.extent.end.line
            if 0 < sl <= len(lines) and 0 < el <= len(lines) and sl <= el:
                extracted = "\n".join(lines[sl - 1 : el]).strip()
        if not extracted:
            # fallback by tokens
            extracted = " ".join(t.spelling for t in node.get_tokens()).strip()
        return extracted

    def _extract_preprocessor_conditions(self, file_path: str) -> None:
        content = self._read_file_content(file_path)
        if not content:
            return
        lines = content.splitlines()
        stack: List[PreprocessorCondition] = []
        current: Optional[PreprocessorCondition] = None
        blocks: List[PreprocessorCondition] = []

        for line_num, raw in enumerate(lines, 1):
            line = raw.strip()
            if line.startswith("#if ") or line.startswith("#ifdef ") or line.startswith("#ifndef "):
                new = PreprocessorCondition(condition=line, start_line=line_num, parent=current)
                blocks.append(new)
                if current:
                    current.children.append(new)
                stack.append(new)
                current = new
            elif line.startswith("#endif"):
                if stack:
                    ended = stack.pop()
                    ended.end_line = line_num
                    current = stack[-1] if stack else None
            elif line.startswith("#elif ") or line.startswith("#else"):
                if current:
                    current.end_line = line_num - 1
                    newb = PreprocessorCondition(condition=line, start_line=line_num, parent=current.parent)
                    blocks.append(newb)
                    if current.parent:
                        current.parent.children.append(newb)
                    stack.pop()
                    stack.append(newb)
                    current = newb

        self.preprocessor_conditions[file_path] = blocks

    def _get_condition_at_line(self, file_path: str, line_num: int) -> str:
        blocks = self.preprocessor_conditions.get(file_path, [])
        active: List[str] = []
        for b in blocks:
            if b.start_line <= line_num and (b.end_line == -1 or b.end_line >= line_num):
                chain = [b.condition]
                cur = b.parent
                while cur:
                    chain.insert(0, cur.condition)
                    cur = cur.parent
                active.append(" -> ".join(chain))
        return "\n".join(active)

    def _extract_macros(self, file_path: str) -> None:
        content = self._read_file_content(file_path)
        if not content:
            return
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#define "):
                start_line_num = i + 1
                condition = self._get_condition_at_line(file_path, start_line_num)
                full_macro = line
                while line.endswith("\\") and i + 1 < len(lines):
                    i += 1
                    line = lines[i].strip()
                    full_macro += "\n" + line
                parts = full_macro.split(None, 2)
                if len(parts) >= 2:
                    macro_name_part = parts[1]
                    macro_value = parts[2] if len(parts) >= 3 else ""
                    is_function_like = False
                    parameters: List[str] = []
                    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", macro_name_part)
                    if m:
                        is_function_like = True
                        macro_name = m.group(1)
                        param_str = m.group(2)
                        parameters = [p.strip() for p in param_str.split(",") if p.strip()]
                    else:
                        macro_name = macro_name_part
                    self.macro_cache[macro_name] = MacroInfo(
                        name=macro_name,
                        value=macro_value,
                        location=f"{file_path}:{start_line_num}",
                        is_constant=not is_function_like,
                        is_function_like=is_function_like,
                        parameters=parameters,
                        definition=full_macro,
                        condition=condition,
                    )
            i += 1

    def _process_struct(self, node) -> Optional[StructInfo]:
        if not node.location.file:
            return None
        struct_name = node.spelling or f"anonymous_struct_{hash(str(node.location))}"
        definition = self._extract_source_code(node)
        is_complete = node.is_definition()
        typedef_name = None
        if node.semantic_parent and node.semantic_parent.kind == CursorKind.TYPEDEF_DECL:
            typedef_name = node.semantic_parent.spelling
        file_path = os.path.abspath(str(node.location.file))
        condition = self._get_condition_at_line(file_path, node.location.line)
        fields: Dict[str, str] = {}
        if is_complete:
            for child in node.get_children():
                if child.kind == CursorKind.FIELD_DECL:
                    fields[child.spelling] = child.type.spelling
        info = StructInfo(
            name=struct_name,
            definition=definition,
            location=str(node.location),
            is_complete=is_complete,
            typedef_name=typedef_name,
            fields=fields,
            condition=condition,
        )
        self.structs[struct_name] = info
        return info

    def _process_global(self, node) -> Optional[GlobalVarInfo]:
        if not node.location.file:
            return None
        file_path = os.path.abspath(str(node.location.file))
        condition = self._get_condition_at_line(file_path, node.location.line)
        is_extern = node.storage_class == StorageClass.EXTERN
        is_static = node.storage_class == StorageClass.STATIC
        type_str = node.type.spelling
        definition = self._extract_source_code(node)
        full_definition = self._extract_source_code(node) if node.is_definition() else None
        initializer = None
        used_macros: Dict[str, MacroInfo] = {}
        if "=" in definition:
            rhs = definition.split("=", 1)[1].strip().rstrip(";")
            initializer = rhs
            for mn, mi in self.macro_cache.items():
                if mn in rhs:
                    used_macros[mn] = mi
        info = GlobalVarInfo(
            name=node.spelling,
            type_str=type_str,
            location=str(node.location),
            is_extern=is_extern,
            is_static=is_static,
            definition=definition,
            initializer=initializer,
            full_definition=full_definition,
            used_macros=used_macros,
            condition=condition,
        )
        self.globals[node.spelling] = info
        return info

    def _add_function(self, node) -> None:
        if not node.location.file:
            return
        if not node.is_definition():
            return
        file_path = os.path.abspath(str(node.extent.start.file))
        if not file_path.startswith(self.code_root):
            return
        func_name = node.spelling
        func_key = f"{func_name}@{str(node.location)}"
        if func_key in self.functions:
            return

        condition = self._get_condition_at_line(file_path, node.location.line)
        return_type = node.result_type.spelling if hasattr(node, "result_type") else ""
        params: List[Dict[str, str]] = []
        try:
            for p in node.get_arguments():
                params.append({"name": p.spelling, "type": p.type.spelling})
        except Exception:
            pass
        src = self._extract_source_code(node)
        self.functions[func_key] = _FuncInternal(
            key=func_key,
            id=self._func_id_counter,
            name=func_name,
            file=file_path,
            location=str(node.location),
            start_line=node.extent.start.line,
            end_line=node.extent.end.line,
            return_type=return_type,
            parameters=params,
            condition=condition,
            source_code=src,
        )
        self._func_id_counter += 1

    def _collect_declarations(self, node) -> None:
        if node.kind == CursorKind.FUNCTION_DECL:
            self._add_function(node)
        elif node.kind == CursorKind.STRUCT_DECL:
            self._process_struct(node)
        elif node.kind == CursorKind.VAR_DECL:
            if node.semantic_parent and node.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
                self._process_global(node)
        elif node.kind == CursorKind.TYPEDEF_DECL:
            try:
                self.typedef_map[node.spelling] = node.underlying_typedef_type.spelling
            except Exception:
                pass
            for ch in node.get_children():
                if ch.kind == CursorKind.STRUCT_DECL:
                    self._process_struct(ch)

        for ch in node.get_children():
            self._collect_declarations(ch)

    def _analyze_function_body(self, node) -> None:
        if not self._current_function_key:
            return
        fn = self.functions[self._current_function_key]

        # tokens-based macro detection (more robust)
        try:
            for tok in node.get_tokens():
                t = tok.spelling
                if t in self.macro_cache:
                    fn.used_macros[t] = self.macro_cache[t]
        except Exception:
            pass

        for child in node.walk_preorder():
            if child.kind == CursorKind.CALL_EXPR:
                try:
                    if child.referenced and child.referenced.location:
                        called_name = child.referenced.spelling
                        fn.calls[called_name] = self._extract_source_code(child.referenced)
                        # add caller edge if we can resolve callee key by name
                        for k, cand in self.functions.items():
                            if cand.name == called_name:
                                cand.callers.add(fn.key)
                                break
                    else:
                        # fallback: store call expression snippet
                        snippet = self._extract_source_code(child)
                        if snippet:
                            fn.calls[snippet] = snippet
                except Exception:
                    continue

            elif child.kind == CursorKind.TYPE_REF:
                try:
                    if child.referenced:
                        tn = child.referenced.spelling
                        if tn in self.structs:
                            fn.used_structs[tn] = self.structs[tn]
                        elif tn in self.typedef_map:
                            orig = self.typedef_map[tn]
                            if orig in self.structs:
                                fn.used_structs[tn] = self.structs[orig]
                            fn.used_typedefs.add(tn)
                except Exception:
                    continue

            elif child.kind == CursorKind.DECL_REF_EXPR:
                try:
                    if child.referenced:
                        rn = child.referenced.spelling
                        if rn in self.globals:
                            fn.used_globals[rn] = self.globals[rn]
                        elif rn in self.macro_cache:
                            fn.used_macros[rn] = self.macro_cache[rn]
                except Exception:
                    continue

    def _analyze_dependencies(self, node) -> None:
        if node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
            func_key = f"{node.spelling}@{str(node.location)}"
            if func_key in self.functions:
                self._current_function_key = func_key
                self._analyze_function_body(node)
                self._current_function_key = None
        for ch in node.get_children():
            self._analyze_dependencies(ch)

    def _parse_file(self, file_path: str) -> None:
        self._extract_preprocessor_conditions(file_path)
        self._extract_macros(file_path)
        try:
            tu = self.index.parse(
                file_path,
                args=self.compile_args,
                options=TranslationUnit.PARSE_INCOMPLETE | TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except Exception:
            return
        self._collect_declarations(tu.cursor)
        self._analyze_dependencies(tu.cursor)

    def analyze(self) -> List[FunctionRecord]:
        for file_path in self._iter_source_files():
            self.processed_files.add(file_path)
            self._parse_file(file_path)
            if len(self.functions) >= self.max_functions:
                break

        # Deterministic function ordering + stable function_id assignment:
        # sort by (file_path, start_line, end_line, function_name, location)
        fns: List[_FuncInternal] = list(self.functions.values())
        fns.sort(
            key=lambda f: (
                os.path.relpath(f.file, self.code_root),
                int(f.start_line or 0),
                int(f.end_line or 0),
                f.name,
                f.location,
            )
        )
        for i, fn in enumerate(fns, 1):
            fn.id = i

        # index by name -> best definition (deterministic: first in sorted list)
        by_name: Dict[str, _FuncInternal] = {}
        for fn in fns:
            if fn.name not in by_name:
                by_name[fn.name] = fn

        out: List[FunctionRecord] = []
        for fn in fns:
            deps = {
                "called_functions": {
                    name: {
                        "definition": (by_name[name].source_code if name in by_name else None),
                    }
                    for name in fn.calls.keys()
                },
                "callers": {
                    caller: {
                        "source_code": (self.functions[caller].source_code if caller in self.functions else None),
                        "file": (self.functions[caller].file if caller in self.functions else None),
                        "location": (self.functions[caller].location if caller in self.functions else None),
                    }
                    for caller in sorted(fn.callers)
                },
                "used_macros": {k: asdict(v) for k, v in fn.used_macros.items()},
                "used_structs": {k: asdict(v) for k, v in fn.used_structs.items()},
                "used_globals": {
                    k: {
                        **asdict(v),
                        "used_macros": {mn: asdict(mi) for mn, mi in v.used_macros.items()},
                    }
                    for k, v in fn.used_globals.items()
                },
                "used_typedefs": sorted(list(fn.used_typedefs)),
            }

            out.append(
                FunctionRecord(
                    function_id=fn.id,
                    function_name=fn.name,
                    file_path=os.path.relpath(fn.file, self.code_root),
                    start_line=fn.start_line,
                    end_line=fn.end_line,
                    source_code=fn.source_code,
                    return_type=fn.return_type,
                    parameters=fn.parameters,
                    condition=fn.condition,
                    dependencies=deps,
                )
            )
        return out


def extract_functions_with_deps(
    code_root: str,
    max_files: int = 2000,
    max_functions: int = 5000,
    libclang_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
    file_patterns: Optional[List[str]] = None,
) -> List[FunctionRecord]:
    analyzer = CodeAnalyzer(
        code_root=code_root,
        libclang_path=libclang_path,
        include_dirs=include_dirs,
        extra_args=extra_args,
        file_patterns=file_patterns,
        max_files=max_files,
        max_functions=max_functions,
    )
    return analyzer.analyze()




