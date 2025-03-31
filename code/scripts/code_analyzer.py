from clang.cindex import Index, CursorKind, TranslationUnit, StorageClass, TokenKind
from dataclasses import dataclass, field, asdict
from typing import Set, Dict, Optional, List, Any, Tuple, Union
from collections import defaultdict
import os
import glob
import json
from datetime import datetime
import csv
import re

# 添加以下导入
import clang.cindex

# 设置 libclang 的路径
clang.cindex.Config.set_library_file('/data/a/ykw/local/clang/lib/libclang.so')

@dataclass
class MacroInfo:
    """宏定义信息类"""
    name: str
    value: str  # 宏定义的值
    location: str
    is_constant: bool = False
    is_function_like: bool = False  # 是否为带参数的函数式宏
    parameters: List[str] = field(default_factory=list)  # 函数宏的参数列表
    definition: str = ""  # 完整的宏定义文本
    condition: str = ""  # 宏定义所在的条件编译块

@dataclass
class PreprocessorCondition:
    """预处理器条件类"""
    condition: str
    start_line: int
    end_line: int = -1
    parent: Optional['PreprocessorCondition'] = None
    children: List['PreprocessorCondition'] = field(default_factory=list)

@dataclass
class GlobalVarInfo:
    """全局变量信息类"""
    name: str
    type_str: str
    location: str
    is_extern: bool
    is_static: bool
    definition: str  # 变量声明
    initializer: Optional[str] = None  # 初始化值
    full_definition: Optional[str] = None  # 完整的定义，包括初始化值
    used_macros: Dict[str, 'MacroInfo'] = field(default_factory=dict)  # 在初始化中使用的宏
    condition: str = ""  # 变量所在的条件编译块

@dataclass
class StructInfo:
    """结构体信息类"""
    name: str
    definition: str
    location: str
    is_complete: bool = False
    typedef_name: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)  # 结构体字段名和类型
    condition: str = ""  # 结构体所在的条件编译块

@dataclass
class Function:
    """函数信息类"""
    name: str
    location: str
    file: str
    is_definition: bool
    id: int = 0                          # 移到必填字段后
    source_code: str = ""
    start_line: int = 0         # 新增：函数起始行号
    end_line: int = 0           # 新增：函数终止行号
    calls: Dict[str, str] = field(default_factory=dict)
    callers: Set[str] = field(default_factory=set)
    used_macros: Dict[str, MacroInfo] = field(default_factory=dict)
    used_structs: Dict[str, StructInfo] = field(default_factory=dict)
    used_globals: Dict[str, GlobalVarInfo] = field(default_factory=dict)
    used_typedefs: Set[str] = field(default_factory=set)
    condition: str = ""  # 函数所在的条件编译块
    parameters: List[Tuple[str, str]] = field(default_factory=list)  # 参数名和类型
    return_type: str = ""  # 返回类型

class CodeAnalyzer:
    def __init__(self):
        self.index = Index.create()
        self.functions: Dict[str, Function] = {}
        self.structs: Dict[str, StructInfo] = {}
        self.globals: Dict[str, GlobalVarInfo] = {}
        self.typedef_map: Dict[str, str] = {}
        self.current_function: Optional[str] = None
        self.processed_files: Set[str] = set()
        self.file_contents: Dict[str, str] = {}
        self.macro_cache: Dict[str, MacroInfo] = {}
        self.preprocessor_conditions: Dict[str, List[PreprocessorCondition]] = defaultdict(list)
        self.main_project_path: str = ""  # 主项目路径
        
        self.compile_args = [
            '-I/usr/include',
            '-I/usr/local/include',
            '-I/usr/lib/gcc/x86_64-linux-gnu/9/include',  # GCC 9 系统头文件
            '-I/usr/include/x86_64-linux-gnu',            # 架构特定头文件
            '-I/lib/x86_64-linux-gnu',                    # 系统库头文件
            '--std=c11',
            '-DUSE_SSL'
        ]

        apr_paths = [
            '/data/a/ykw/build/httpd-2.4.62/srclib/apr',      # APR库源码
            '/data/a/ykw/build/httpd-2.4.62/srclib/apr-util',   # APR-util库源码
        ]
    
        for path in apr_paths:
            if os.path.exists(path):
                print(f"Found APR source at: {path}")
                self.add_include_path(path)
                # 递归搜索所有.c文件
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith('.c'):
                            file_path = os.path.join(root, file)
                            if file_path not in self.processed_files:
                                print(f"Adding APR source file: {file_path}")
                                self.processed_files.add(file_path)
        # 初始化函数 id 计数器
        self.function_id_counter = 1

    def add_include_path(self, path: str) -> None:
        """添加头文件搜索路径"""
        if os.path.exists(path):
            self.compile_args.append(f'-I{path}')
        else:
            print(f"Warning: Include path does not exist: {path}")

    def _read_file_content(self, file_path: str) -> str:
        """读取文件内容并缓存"""
        if (file_path not in self.file_contents):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_contents[file_path] = f.read()
            except UnicodeDecodeError:
                # 如果utf-8失败，尝试以二进制方式读取
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        self.file_contents[file_path] = content.decode('latin-1')
                except Exception as e:
                    print(f"Error reading {file_path} with latin-1: {e}")
                    return ""
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return ""
        return self.file_contents[file_path]

    def _extract_source_code(self, node) -> str:
        """提取节点对应的源代码，如果直接截取为空则尝试通过行号和 tokens 重构"""
        if not node.extent.start.file:
            return ""
        
        file_path = str(node.extent.start.file)
        content = self._read_file_content(file_path)
        
        start_offset = node.extent.start.offset
        end_offset = node.extent.end.offset

        extracted_code = ""
        if content and start_offset >= 0 and end_offset >= 0 and start_offset < len(content) and end_offset <= len(content):
            extracted_code = content[start_offset:end_offset].strip()

        # 备选方案1：利用起始行和终止行（行号从 1 开始，所以需要 -1）
        if not extracted_code and content:
            lines = content.splitlines()
            start_line = node.extent.start.line
            end_line = node.extent.end.line
            if 0 < start_line <= len(lines) and 0 < end_line <= len(lines):
                extracted_code = "\n".join(lines[start_line - 1:end_line]).strip()
        
        # 备选方案2：利用 token 重构
        if not extracted_code:
            tokens = list(node.get_tokens())
            extracted_code = " ".join(token.spelling for token in tokens).strip()

        return extracted_code

    def _extract_preprocessor_conditions(self, file_path: str) -> None:
        """提取文件中的预处理器条件块"""
        content = self._read_file_content(file_path)
        if not content:
            return
            
        lines = content.splitlines()
        condition_stack = []
        current_condition = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 处理条件开始
            if line.startswith("#if ") or line.startswith("#ifdef ") or line.startswith("#ifndef "):
                condition = line
                
                new_condition = PreprocessorCondition(
                    condition=condition,
                    start_line=line_num,
                    parent=current_condition
                )
                
                self.preprocessor_conditions[file_path].append(new_condition)
                
                if current_condition:
                    current_condition.children.append(new_condition)
                    
                condition_stack.append(new_condition)
                current_condition = new_condition
                
            # 处理条件结束
            elif line.startswith("#endif"):
                if condition_stack:
                    ended_condition = condition_stack.pop()
                    ended_condition.end_line = line_num
                    
                    if condition_stack:
                        current_condition = condition_stack[-1]
                    else:
                        current_condition = None
                        
            # 处理条件分支
            elif line.startswith("#elif ") or line.startswith("#else"):
                if current_condition:
                    # 为原条件标记结束
                    current_condition.end_line = line_num - 1
                    
                    # 创建新的条件分支
                    new_branch = PreprocessorCondition(
                        condition=line,
                        start_line=line_num,
                        parent=current_condition.parent
                    )
                    
                    self.preprocessor_conditions[file_path].append(new_branch)
                    
                    if current_condition.parent:
                        current_condition.parent.children.append(new_branch)
                        
                    # 替换栈顶元素
                    condition_stack.pop()
                    condition_stack.append(new_branch)
                    current_condition = new_branch

    def _get_condition_at_line(self, file_path: str, line_num: int) -> str:
        """获取指定行所在的预处理器条件"""
        if file_path not in self.preprocessor_conditions:
            return ""
            
        active_conditions = []
        
        for condition in self.preprocessor_conditions[file_path]:
            if condition.start_line <= line_num and (condition.end_line == -1 or condition.end_line >= line_num):
                # 递归检查父条件
                current = condition
                condition_chain = [current.condition]
                
                while current.parent:
                    current = current.parent
                    condition_chain.insert(0, current.condition)
                    
                active_conditions.append(" -> ".join(condition_chain))
                
        return "\n".join(active_conditions)

    def _extract_macros(self, file_path: str) -> None:
        """提取文件中的所有宏定义，包括多行宏和函数式宏"""
        content = self._read_file_content(file_path)
        if not content:
            return
            
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检查宏定义
            if line.startswith("#define "):
                start_line_num = i + 1
                
                # 获取当前行所在的条件
                condition = self._get_condition_at_line(file_path, start_line_num)
                
                # 处理多行宏
                full_macro = line
                while line.endswith("\\") and i + 1 < len(lines):
                    i += 1
                    line = lines[i].strip()
                    full_macro += "\n" + line
                
                # 解析宏定义
                macro_parts = full_macro.split(None, 2)  # 最多分成3部分
                
                if len(macro_parts) >= 2:
                    macro_name_part = macro_parts[1]
                    macro_value = ""
                    if len(macro_parts) >= 3:
                        macro_value = macro_parts[2]
                    
                    # 检查是否是函数式宏
                    is_function_like = False
                    parameters = []
                    
                    # 使用正则表达式匹配函数式宏
                    func_macro_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)', macro_name_part)
                    if func_macro_match:
                        is_function_like = True
                        macro_name = func_macro_match.group(1)
                        param_str = func_macro_match.group(2)
                        # 解析参数列表
                        parameters = [p.strip() for p in param_str.split(',') if p.strip()]
                    else:
                        macro_name = macro_name_part
                    
                    # 存储宏信息
                    self.macro_cache[macro_name] = MacroInfo(
                        name=macro_name,
                        value=macro_value,
                        location=f"{file_path}:{start_line_num}",
                        is_constant=not is_function_like,
                        is_function_like=is_function_like,
                        parameters=parameters,
                        definition=full_macro,
                        condition=condition
                    )
            
            i += 1

    def _extract_array_initializer(self, node):
        """手动提取数组初始化的完整文本"""
        if not node.location.file:
            return ""

        file_path = str(node.location.file)
        content = self._read_file_content(file_path)

        start_offset = node.extent.start.offset
        end_offset = node.extent.end.offset

        if not content or start_offset < 0 or end_offset < 0 or start_offset >= len(content) or end_offset > len(content):
            return ""

        init_text = content[start_offset:end_offset].strip()
        return init_text

    def _expand_macros(self, text, already_expanded=None):
        """递归展开宏引用，防止循环展开"""
        if already_expanded is None:
            already_expanded = set()
            
        for macro_name, macro_info in self.macro_cache.items():
            if macro_name in already_expanded:
                continue
                
            if macro_name in text and not macro_info.is_function_like:
                already_expanded.add(macro_name)
                expanded = text.replace(macro_name, macro_info.value)
                # 递归展开
                return self._expand_macros(expanded, already_expanded)
                
        return text

    def _process_struct_definition(self, node) -> Optional[StructInfo]:
        """处理结构体定义，提取字段信息"""
        if not node.location.file:
            return None

        # 获取结构体名称
        struct_name = node.spelling or f"anonymous_struct_{hash(node.location)}"
        
        # 提取完整的结构体定义
        definition = self._extract_source_code(node)
        
        # 检查是否是完整的结构体定义
        is_complete = node.is_definition()
        
        # 检查是否有 typedef 名称
        typedef_name = None
        if node.semantic_parent and node.semantic_parent.kind == CursorKind.TYPEDEF_DECL:
            typedef_name = node.semantic_parent.spelling
        
        # 获取所在条件
        condition = ""
        if node.location.file:
            condition = self._get_condition_at_line(str(node.location.file), node.location.line)
        
        # 提取字段信息
        fields = {}
        if is_complete:
            for child in node.get_children():
                if child.kind == CursorKind.FIELD_DECL:
                    field_name = child.spelling
                    field_type = child.type.spelling
                    fields[field_name] = field_type
        
        struct_info = StructInfo(
            name=struct_name,
            definition=definition,
            location=str(node.location),
            is_complete=is_complete,
            typedef_name=typedef_name,
            fields=fields,
            condition=condition
        )
        
        self.structs[struct_name] = struct_info
        return struct_info

    def _process_global_variable(self, node) -> Optional[GlobalVarInfo]:
        """处理全局变量声明和定义，支持处理数组初始化中的宏定义"""
        if not node.location.file:
            return None

        # 基本信息
        is_extern = node.storage_class == StorageClass.EXTERN
        is_static = node.storage_class == StorageClass.STATIC
        type_str = node.type.spelling

        # 获取变量声明
        definition = self._extract_source_code(node)

        # 获取完整的定义（包括数组初始化内容等）
        full_definition = None
        if node.is_definition():
            full_definition = self._extract_source_code(node)

        # 获取所在条件
        condition = ""
        if node.location.file:
            condition = self._get_condition_at_line(str(node.location.file), node.location.line)

        # 获取初始化值（如果有）并识别使用的宏
        initializer_str = None
        used_macros = {}
        
        # 处理初始化表达式
        for child in node.get_children():
            if child.kind in [CursorKind.INIT_LIST_EXPR, CursorKind.UNEXPOSED_EXPR]:
                # 提取完整初始化表达式
                init_text = self._extract_array_initializer(child)
                
                if init_text:
                    initializer_str = init_text
                    
                    # 识别初始化中使用的宏
                    for macro_name, macro_info in self.macro_cache.items():
                        if macro_name in init_text:
                            used_macros[macro_name] = macro_info
        
        # 如果是简单的赋值表达式，直接获取右侧的值
        if initializer_str is None and "=" in definition:
            parts = definition.split("=", 1)
            if len(parts) == 2:
                initializer_str = parts[1].strip().rstrip(";")
                
                # 识别初始化中使用的宏
                for macro_name, macro_info in self.macro_cache.items():
                    if macro_name in initializer_str:
                        used_macros[macro_name] = macro_info

        global_info = GlobalVarInfo(
            name=node.spelling,
            type_str=type_str,
            location=str(node.location),
            is_extern=is_extern,
            is_static=is_static,
            definition=definition,
            initializer=initializer_str,
            full_definition=full_definition,
            used_macros=used_macros,
            condition=condition
        )

        self.globals[node.spelling] = global_info
        return global_info
    
    def _add_function(self, node) -> None:
        """添加函数声明及其源代码"""
        if not node.location.file:
            return

        file_path = str(node.extent.start.file)
        is_definition = node.is_definition()
        source_code = self._extract_source_code(node) if is_definition else ""
        start_line = node.extent.start.line if node.extent.start else 0
        end_line = node.extent.end.line if node.extent.end else 0
        
        # 获取返回类型
        return_type = node.result_type.spelling
        
        # 获取参数列表
        parameters = []
        for param in node.get_arguments():
            param_name = param.spelling
            param_type = param.type.spelling
            parameters.append((param_name, param_type))
        
        # 获取所在条件
        condition = ""
        if node.location.file:
            condition = self._get_condition_at_line(str(node.location.file), node.location.line)

        # Use full location for uniqueness (includes line/column)
        func_key = f"{node.spelling}@{str(node.location)}"
        
        if func_key in self.functions:
            if is_definition:
                func = self.functions[func_key]
                func.is_definition = True
                func.file = file_path
                func.location = str(node.location)
                func.source_code = source_code
                func.start_line = start_line
                func.end_line = end_line
                func.parameters = parameters
                func.return_type = return_type
                func.condition = condition
        else:
            self.functions[func_key] = Function(
                id=self.function_id_counter,     # 赋予唯一 id
                name=node.spelling,
                location=str(node.location),
                file=file_path,
                is_definition=is_definition,
                source_code=source_code,
                start_line=start_line,
                end_line=end_line,
                parameters=parameters,
                return_type=return_type,
                condition=condition
            )
            self.function_id_counter += 1

    def _collect_declarations(self, node) -> None:
        """收集所有声明和定义"""
        if not node.location.file:
            pass
        elif node.kind == CursorKind.FUNCTION_DECL:
            self._add_function(node)
        elif node.kind == CursorKind.STRUCT_DECL:
            self._process_struct_definition(node)
        elif node.kind == CursorKind.VAR_DECL:
            if node.semantic_parent and node.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
                self._process_global_variable(node)
        elif node.kind == CursorKind.TYPEDEF_DECL:
            underlying_type = node.underlying_typedef_type.spelling
            self.typedef_map[node.spelling] = underlying_type
            
            for child in node.get_children():
                if child.kind == CursorKind.STRUCT_DECL:
                    self._process_struct_definition(child)
        elif node.kind == CursorKind.ENUM_DECL:
            # 处理枚举类型
            enum_name = node.spelling or f"anonymous_enum_{hash(node.location)}"
            enum_definition = self._extract_source_code(node)
            
            # 获取所在条件
            condition = ""
            if node.location.file:
                condition = self._get_condition_at_line(str(node.location.file), node.location.line)
            
            # 处理枚举常量
            for child in node.get_children():
                if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                    const_name = child.spelling
                    const_value = child.enum_value
                    
                    # 将枚举常量存为宏常量
                    self.macro_cache[const_name] = MacroInfo(
                        name=const_name,
                        value=str(const_value),
                        location=str(child.location),
                        is_constant=True,
                        definition=f"enum {enum_name} {const_name} = {const_value}",
                        condition=condition
                    )

        for child in node.get_children():
            self._collect_declarations(child)

    def _analyze_function_body(self, node) -> None:
        """分析函数体中的依赖"""
        if not self.current_function:
            return

        for child in node.walk_preorder():
            if child.kind == CursorKind.UNEXPOSED_EXPR:
                # 检查是否是宏引用
                tokens = list(child.get_tokens())
                for token in tokens:
                    macro_name = token.spelling
                    if macro_name in self.macro_cache:
                        self.functions[self.current_function].used_macros[macro_name] = \
                            self.macro_cache[macro_name]
            
            elif child.kind == CursorKind.CALL_EXPR:
                if child.referenced and child.referenced.location:
                    # 使用详细地点构造完整键
                    called_func_name = child.referenced.spelling
                    called_func_key = f"{called_func_name}@{str(child.referenced.location)}"
                    
                    # 也尝试使用简单名称查找
                    func_keys = [k for k in self.functions.keys() if k.startswith(f"{called_func_name}@")]
                    
                    if called_func_key in self.functions:
                        called_func_source = self._extract_source_code(child.referenced)
                        self.functions[self.current_function].calls[called_func_name] = called_func_source
                        self.functions[called_func_key].callers.add(self.current_function)
                    elif func_keys:
                        # 使用第一个匹配的函数
                        best_key = func_keys[0]
                        called_func_source = self._extract_source_code(child.referenced) or self.functions[best_key].source_code
                        self.functions[self.current_function].calls[called_func_name] = called_func_source
                        self.functions[best_key].callers.add(self.current_function)
                    else:
                        # 处理间接调用
                        called_text = self._extract_source_code(child)
                        self.functions[self.current_function].calls[called_func_name] = called_text
            
            elif child.kind == CursorKind.TYPE_REF:
                # 处理类型引用（结构体等）
                if child.referenced:
                    type_name = child.referenced.spelling
                    if type_name in self.structs:
                        self.functions[self.current_function].used_structs[type_name] = \
                            self.structs[type_name]
                    elif type_name in self.typedef_map:
                        original_type = self.typedef_map[type_name]
                        if original_type in self.structs:
                            self.functions[self.current_function].used_structs[type_name] = \
                                self.structs[original_type]
                        self.functions[self.current_function].used_typedefs.add(type_name)
            
            elif child.kind == CursorKind.DECL_REF_EXPR:
                # 处理变量引用
                if child.referenced:
                    ref_name = child.referenced.spelling
                    # 检查是否是全局变量的引用
                    if ref_name in self.globals:
                        self.functions[self.current_function].used_globals[ref_name] = \
                            self.globals[ref_name]
                    # 检查是否是宏引用
                    elif ref_name in self.macro_cache:
                        self.functions[self.current_function].used_macros[ref_name] = \
                            self.macro_cache[ref_name]

    def _analyze_dependencies(self, node) -> None:
        """分析依赖关系"""
        if not node.location.file:
            pass
        elif node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
            func_key = f"{node.spelling}@{str(node.location)}"
            self.current_function = func_key
            self._analyze_function_body(node)
            self.current_function = None
        
        for child in node.get_children():
            self._analyze_dependencies(child)
            
    def parse_project(self, project_paths: List[str], file_patterns: List[str] = None) -> None:
        """解析项目目录，只分析主目录源文件但保持对APR的依赖分析"""
        if file_patterns is None:
            file_patterns = ['*.c']
        
        # 只分析主目录的源文件
        self.main_project_path = project_paths[0]  # 假设第一个路径是主项目路径
        
        for pattern in file_patterns:
            full_pattern = os.path.join(self.main_project_path, '**', pattern)
            for file_path in glob.glob(full_pattern, recursive=True):
                if file_path not in self.processed_files:
                    print(f"Processing file: {file_path}")
                    self.parse_file(file_path)
                    self.processed_files.add(file_path)

    def parse_file(self, filename: str) -> None:
        """解析单个文件"""
        # 提取预处理器条件
        self._extract_preprocessor_conditions(filename)
        
        # 提取宏定义
        self._extract_macros(filename)
        
        try:
            tu = self.index.parse(
                filename,
                args=self.compile_args,
                options=TranslationUnit.PARSE_INCOMPLETE | 
                        TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            for diag in tu.diagnostics:
                if diag.severity >= 3:  # Error or Fatal
                    print(f"Error in {filename}: {diag.spelling}")
                else:  # Warning or Note
                    print(f"Warning in {filename}: {diag.spelling}")
            
            self._collect_declarations(tu.cursor)
            self._analyze_dependencies(tu.cursor)
            
        except Exception as e:
            print(f"Failed to parse {filename}: {str(e)}")

    def export_to_json(self, output_file: str, 
                         include_metrics: bool = True,
                         include_source: bool = True,
                         pretty_print: bool = True) -> None:
        """导出分析结果到JSON文件"""
        def is_main_project_file(path: str) -> bool:
            """检查文件是否属于主项目目录"""
            return path.startswith(self.main_project_path)

        result = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_files": len([f for f in self.processed_files if is_main_project_file(f)]),
                "num_functions": len([f for f in self.functions.values() 
                                    if f.is_definition and is_main_project_file(f.file)]),
                "files_analyzed": sorted([f for f in self.processed_files if is_main_project_file(f)])
            },
            "functions": {},
            "structs": {},
            "globals": {},
            "macros": {}
        }
        
        # 同时准备CSV导出
        csv_file = output_file.replace('.json', '.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "function_id", "function_name", "file_path",
                "start_line", "end_line", "return_type", "parameters",
                "dependencies", "source_code", "condition"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # 导出函数信息
            function_count = 0
            for func_name, func in self.functions.items():
                if not func.is_definition or func.file.endswith('.h'):
                    continue
                # 跳过不在主项目路径下的函数
                if not is_main_project_file(func.file):
                    continue
                        
                function_data = {
                    "basic_info": {
                        "function_id": func.id,
                        "name": func.name,
                        "file": func.file,
                        "location": func.location,
                        "is_definition": func.is_definition,
                        "start_line": func.start_line,
                        "end_line": func.end_line,
                        "return_type": func.return_type,
                        "parameters": [{"name": name, "type": type_str} for name, type_str in func.parameters],
                        "condition": func.condition
                    },
                    "dependencies": {
                        "called_functions": {
                            name: {
                                "declaration": source,
                                "is_external": not any(
                                    called_func.file.startswith(self.main_project_path)
                                    for called_func in self.functions.values() 
                                    if called_func.name == name and called_func.is_definition
                                ),
                                "definition": next(
                                    (f.source_code for f in self.functions.values() 
                                     if f.name == name and f.is_definition and is_main_project_file(f.file)),
                                    None
                                )
                            }
                            for name, source in func.calls.items()
                        },
                        "callers": {
                            caller: {
                                "source_code": self.functions[caller].source_code if caller in self.functions else None,
                                "file": self.functions[caller].file if caller in self.functions else None,
                                "location": self.functions[caller].location if caller in self.functions else None,
                            }
                            for caller in sorted(func.callers)
                        },
                        "used_macros": {
                            name: {
                                "name": info.name,
                                "value": info.value,
                                "is_constant": info.is_constant,
                                "is_function_like": info.is_function_like,
                                "parameters": info.parameters,
                                "definition": info.definition,
                                "location": info.location,
                                "condition": info.condition
                            }
                            for name, info in func.used_macros.items()
                        },
                        "used_structs": {
                            name: {
                                "name": info.name,
                                "definition": info.definition,
                                "typedef_name": info.typedef_name,
                                "fields": info.fields,
                                "condition": info.condition
                            }
                            for name, info in func.used_structs.items()
                        },
                        "used_globals": {
                            name: {
                                "name": info.name,
                                "type": info.type_str,
                                "is_extern": info.is_extern,
                                "is_static": info.is_static,
                                "definition": info.definition,
                                "initializer": info.initializer,
                                "full_definition": info.full_definition,
                                "used_macros": {
                                    macro_name: {
                                        "name": macro_info.name,
                                        "value": macro_info.value,
                                        "definition": macro_info.definition
                                    }
                                    for macro_name, macro_info in info.used_macros.items()
                                },
                                "condition": info.condition
                            }
                            for name, info in func.used_globals.items()
                        },
                        "used_typedefs": list(func.used_typedefs)
                    }
                }
                function_data["source_code"] = ""
                if include_source and func.source_code:
                    function_data["source_code"] = func.source_code
                        
                result["functions"][func_name] = function_data
                
                # 同时写入CSV
                writer.writerow({
                    "function_id": func.id,
                    "function_name": func.name,
                    "file_path": func.file,
                    "start_line": func.start_line,
                    "end_line": func.end_line,
                    "return_type": func.return_type,
                    "parameters": json.dumps([{"name": name, "type": type_str} for name, type_str in func.parameters]),
                    "dependencies": json.dumps(function_data["dependencies"], ensure_ascii=False),
                    "source_code": func.source_code if include_source else "",
                    "condition": func.condition
                })
                function_count += 1
            
            print(f"Exported {function_count} functions to CSV: {csv_file}")
            
        filtered_structs = {
            name: info for name, info in self.structs.items()
            if is_main_project_file(str(info.location))
        }
        filtered_globals = {
            name: info for name, info in self.globals.items()
            if is_main_project_file(str(info.location))
        }
        filtered_macros = {
            name: info for name, info in self.macro_cache.items()
            if is_main_project_file(str(info.location))
        }
            
        result["structs"] = {
            name: {
                "name": info.name,
                "definition": info.definition,
                "location": info.location,
                "is_complete": info.is_complete,
                "typedef_name": info.typedef_name,
                "fields": info.fields,
                "condition": info.condition
            }
            for name, info in filtered_structs.items()
        }
            
        result["globals"] = {
            name: {
                "name": info.name,
                "type": info.type_str,
                "location": info.location,
                "is_extern": info.is_extern,
                "is_static": info.is_static,
                "definition": info.definition,
                "initializer": info.initializer,
                "full_definition": info.full_definition,
                "used_macros": {
                    macro_name: {
                        "name": macro_info.name,
                        "value": macro_info.value,
                        "definition": macro_info.definition
                    }
                    for macro_name, macro_info in info.used_macros.items()
                },
                "condition": info.condition
            }
            for name, info in filtered_globals.items()
        }
        
        result["macros"] = {
            name: {
                "name": info.name,
                "value": info.value,
                "location": info.location,
                "is_constant": info.is_constant,
                "is_function_like": info.is_function_like,
                "parameters": info.parameters,
                "definition": info.definition,
                "condition": info.condition
            }
            for name, info in filtered_macros.items()
        }
            
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False) 
                
        print(f"Exported {function_count} functions to JSON: {output_file}")

def main_nginx():
    analyzer = CodeAnalyzer()
    nginx_path = "/data/a/ykw/RFC/final/data/nginx/source_code"
    # 添加所有必要的头文件路径
    # 以下是httpd项目的头文件路径示例，可以根据实际项目更改
    include_paths = [
        '/data/a/ykw/RFC/final/data/nginx/source_code/src',
        '/data/a/ykw/RFC/final/data/nginx/source_code/src/core',
        '/data/a/ykw/RFC/final/data/nginx/source_code/src/event',
        '/data/a/ykw/RFC/final/data/nginx/source_code/src/http',
        '/data/a/ykw/RFC/final/data/nginx/source_code/src/os',
    ]
    for path in include_paths:
        analyzer.add_include_path(path)
    
    # 设置主项目路径
    main_project_path = '/data/a/ykw/RFC/final/data/nginx/source_code'
    analyzer.parse_project([main_project_path], ['*.c', '*.h'])
    
    # 导出结果 - 现在会同时生成JSON和CSV
    analyzer.export_to_json(
        output_file='../data/nginx/func/all_func.json',
        include_metrics=True,
        pretty_print=True
    )
    
def main_httpd():
    analyzer = CodeAnalyzer()
    httpd_path = "/data/a/ykw/RFC/final/data/httpd/source_code/modules/http"
    # 添加所有必要的头文件路径
    # 以下是httpd项目的头文件路径示例，可以根据实际项目更改
    include_paths = [
        '/data/a/ykw/httpd/include',
        '/data/a/ykw/RFC/final/data/httpd/source_code/include',
        '/data/a/ykw/RFC/final/data/httpd/source_code/os/unix',
        '/data/a/ykw/RFC/final/data/httpd/source_code/modules/http',
        # APR 相关头文件路径
        '/data/a/ykw/build/httpd-2.4.62/srclib/apr/include',
        '/data/a/ykw/build/httpd-2.4.62/srclib-util/include',
    ]
    
    for path in include_paths:
        analyzer.add_include_path(path)
    
    # 只解析主目录的源文件（此处示例解析 /data/a/ykw/RFC/final/data/httpd/source_code/support 目录下的文件）
    main_project_path = '/data/a/ykw/RFC/final/data/httpd/source_code'
    analyzer.parse_project([main_project_path], ['*.c', '*.h'])
    
    # 导出结果 - 现在会同时生成JSON和CSV
    analyzer.export_to_json(
        output_file='../data/httpd/func/all_func.json',
        include_metrics=True,
        pretty_print=True
    )
    

def main_openssl():
    analyzer = CodeAnalyzer()
    # httpd_path = "/data/a/ykw/RFC/final/data/openssl/source_code
    # 添加所有必要的头文件路径
    # 以下是httpd项目的头文件路径示例，可以根据实际项目更改
    include_paths = [
        "/data/a/ykw/RFC/final/data/openssl/source_code/include",
    ]
    
    for path in include_paths:
        analyzer.add_include_path(path)
    
    # 只解析主目录的源文件（此处示例解析 /data/a/ykw/RFC/final/data/httpd/source_code/support 目录下的文件）
    main_project_path = '/data/a/ykw/RFC/final/data/openssl/source_code/ssl'
    analyzer.parse_project([main_project_path], ['*.c', '*.h'])
    
    # 导出结果 - 现在会同时生成JSON和CSV
    analyzer.export_to_json(
        output_file='../data/openssl/func/all_func.json',
        include_metrics=True,
        pretty_print=True
    )
    
def main_boringssl():
    analyzer = CodeAnalyzer()
    httpd_path = "/data/a/ykw/RFC/final/data/boringssl/source_code"
    # 添加所有必要的头文件路径
    # 以下是httpd项目的头文件路径示例，可以根据实际项目更改
    include_paths = [
        "/data/a/ykw/RFC/final/data/boringssl/source_code/include/openssl",
    ]
    
    for path in include_paths:
        analyzer.add_include_path(path)
    
    # 只解析主目录的源文件（此处示例解析 /data/a/ykw/RFC/final/data/httpd/source_code/support 目录下的文件）
    main_project_path = "/data/a/ykw/RFC/final/data/boringssl/source_code/ssl"
    analyzer.parse_project([main_project_path], ['*.c', '*.h'])
    
    # 导出结果 - 现在会同时生成JSON和CSV
    analyzer.export_to_json(
        output_file='../data/boringssl/func/all_func.json',
        include_metrics=True,
        pretty_print=True
    )

if __name__ == "__main__":
    # main()
    # main_httpd()
    # main_openssl()
    # main_boringssl()
    main_nginx()