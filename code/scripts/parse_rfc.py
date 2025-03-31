import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class RFCParser:
    def __init__(self):
        self.section_pattern = re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)$')
        self.sentence_pattern = re.compile(r'(?<=\.)\s{2,}|\.\n')
        self.para_indent = '   '  # 正好3个空格的缩进
        self.format_start = re.compile(r'^\s*(?:Preferred format:|[A-Za-z-]+ = |\+={3,}|\|)')  # 匹配格式定义、表格开始
        self.format_marker = "Preferred format:"
        self.table_marker = "+====="
        # 匹配"   1. "或"   12. "这样的枚举项
        self.enum_item_pattern = re.compile(r'^\s{3}(?:\d{1,2})\.\s+')
        
        # Keywords for SR extraction
        self.keywords = [
            "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
            "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED",
            "MAY", "OPTIONAL"
        ]

    def is_section_header(self, line: str) -> tuple:
        """检查是否是章节标题 - 章节标题必须从行首开始，不能有空格"""
        if line.startswith(' '):  # 如果行首有空格，一定不是章节标题
            return False, None, None
            
        match = self.section_pattern.match(line.strip())
        if match:
            section_num = match.group(1)
            section_title = match.group(2)
            return True, section_num, section_title
        return False, None, None

    def is_format_para(self, lines: List[str]) -> bool:
        if not lines:
            return False
        # 检查是否是格式定义或表格
        first_line = lines[0].rstrip()
        return (first_line.startswith('     ') or  # 5空格缩进
                bool(self.format_start.match(first_line)) or  # 格式定义开始
                'Preferred format:' in first_line)  # 格式标题

    def split_sentences(self, text: str) -> List[str]:
        parts = self.sentence_pattern.split(text)
        return [part.strip() for part in parts if part.strip()]

    def is_exact_para_indent(self, line: str) -> bool:
        """检查是否恰好是3个空格的缩进"""
        return line.startswith(self.para_indent) and not line.startswith(self.para_indent + ' ')

    def is_format_start(self, line: str) -> bool:
        """检查是否是格式定义的开始"""
        return (line.strip() == self.format_marker or 
                bool(re.match(r'\s{5,}[A-Za-z-]+\s*=', line)))

    def is_table_start(self, line: str) -> bool:
        """检查是否是表格的开始"""
        return self.table_marker in line

    def is_enum_item(self, line: str) -> bool:
        """检查是否是枚举项"""
        return bool(self.enum_item_pattern.match(line))

    def parse_paragraphs(self, lines: List[str]) -> List[Dict[str, Any]]:
        paragraphs = []
        current_para = []
        looking_for_para_start = True
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # 如果是枚举项且当前已有段落，直接加入当前段落
            if current_para and self.is_enum_item(line):
                current_para.append(line)
                i += 1
                continue
                
            # 空行处理
            if not line:
                if current_para:
                    # 看看下一个非空行
                    next_idx = i + 1
                    while next_idx < len(lines) and not lines[next_idx].strip():
                        next_idx += 1
                    
                    # 如果下一行不是枚举项，则结束当前段落
                    if next_idx >= len(lines) or not self.is_enum_item(lines[next_idx]):
                        text = '\n'.join(current_para)
                        paragraphs.append({
                            'content': text,
                            'sentences': self.split_sentences(text)
                        })
                        current_para = []
                        looking_for_para_start = True
                i += 1
                continue
            
            # 寻找新段落的开始
            if looking_for_para_start:
                if self.is_exact_para_indent(line):
                    current_para = [line]
                    looking_for_para_start = False
            else:
                # 在段落中，所有行（无论缩进多少）都属于当前段落
                current_para.append(line)
            
            i += 1
        
        # 处理最后一个段落
        if current_para:
            text = '\n'.join(current_para)
            paragraphs.append({
                'content': text,
                'sentences': self.split_sentences(text)
            })
        
        return paragraphs

    def parse_document(self, content: str) -> Dict[str, Any]:
        lines = content.splitlines()
        root_structure = {'sections': []}
        section_stack = []
        current_content = []
        level4_content = []  # 存储四级标题相关内容

        def handle_current_content():
            if not current_content or not section_stack:
                return
            
            # 处理四级标题的特殊情况
            if level4_content:
                # 找到第一个段落的结束位置
                first_para_end = 0
                empty_line_found = False
                for i, line in enumerate(current_content):
                    if not line.strip():
                        if empty_line_found:  # 第二个空行，标志第一段结束
                            first_para_end = i
                            break
                        empty_line_found = True
                
                if first_para_end == 0:  # 如果没有找到第二个空行，全部内容属于第一段
                    first_para_end = len(current_content)
                
                # 合并四级标题和第一段
                merged_first_para = level4_content + current_content[:first_para_end]
                remaining_content = current_content[first_para_end:]
                
                # 解析段落
                all_paragraphs = []
                if merged_first_para:
                    first_para = self.parse_paragraphs(merged_first_para)
                    all_paragraphs.extend(first_para)
                if remaining_content:
                    other_paras = self.parse_paragraphs(remaining_content)
                    all_paragraphs.extend(other_paras)
                    
                section_stack[-1]['paragraphs'] = all_paragraphs
            else:
                section_stack[-1]['paragraphs'] = self.parse_paragraphs(current_content)

        for line in lines:
            is_section, num, title = self.is_section_header(line)
            if is_section:
                level = len(num.split('.'))
                
                if level == 4:
                    # 收集四级标题内容
                    level4_content = [line]
                    continue
                else:
                    # 处理之前的内容
                    handle_current_content()
                    level4_content = []
                
                # 创建新的section（最多到三级）
                new_section = {
                    'number': num,
                    'title': title,
                    'level': min(level, 3),
                    'paragraphs': [],
                    'sections': []
                }

                # 根据层级关系确定放置位置
                if not section_stack:
                    root_structure['sections'].append(new_section)
                    section_stack = [new_section]
                else:
                    while (section_stack and 
                           section_stack[-1]['level'] >= min(level, 3)):
                        section_stack.pop()
                    
                    if not section_stack:
                        root_structure['sections'].append(new_section)
                    else:
                        section_stack[-1]['sections'].append(new_section)
                    
                    section_stack.append(new_section)
                
                current_content = []
            else:
                current_content.append(line)

        # 处理最后的内容
        handle_current_content()
        
        return root_structure

    def save_to_jsonl(self, structure: Dict[str, Any], output_path: str, pretty: bool = False) -> None:
        """将解析结果保存为JSONL格式
        Args:
            structure: 解析后的文档结构
            output_path: 输出文件路径
            pretty: 是否格式化输出以便于阅读
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        def process_section(section):
            results = []
            # 创建包含所有段落的section记录
            json_obj = {
                'section': {
                    'number': section['number'],
                    'title': section['title'],
                    'level': section['level']
                },
                'paragraphs': [
                    {
                        'index': para_idx,
                        'content': para['content'],
                        'sentences': [
                            {
                                'index': sent_idx,
                                'content': sentence
                            }
                            for sent_idx, sentence in enumerate(para['sentences'], 1)
                        ]
                    }
                    for para_idx, para in enumerate(section['paragraphs'], 1)
                ]
            }
            results.append(json_obj)
            
            # 递归处理子section
            for subsection in section['sections']:
                results.extend(process_section(subsection))
            
            return results

        # 处理所有sections并写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for section in structure['sections']:
                for json_obj in process_section(section):
                    if pretty:
                        json.dump(json_obj, f, ensure_ascii=False, indent=2)
                        f.write('\n\n')  # 额外的空行分隔每个记录
                    else:
                        json.dump(json_obj, f, ensure_ascii=False)
                        f.write('\n')
                        
    def clean_sentence(self, content: str) -> str:
        """清理句子中的换行和多余空格"""
        # 替换换行+三空格的模式
        content = content.replace('\n   ', ' ')
        # 替换普通换行
        content = content.replace('\n', ' ')
        # 将多个空格替换为单个空格
        content = ' '.join(content.split())
        # 清理连字符后的空格
        content = content.replace('- ', '-')
        return content
        
    def extract_sentences_with_keywords(self, jsonl_path: str, output_jsonl_path: str, output_csv_path: str) -> None:
        """提取包含关键词的句子并保存为JSONL和CSV格式
        
        Args:
            jsonl_path: 输入的JSONL文件路径
            output_jsonl_path: 输出的JSONL文件路径
            output_csv_path: 输出的CSV文件路径
        """
        results = []
        
        # 读取整个文件内容
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按双换行符分割JSON对象
        json_objects = content.split('\n\n')
        
        # 创建所有段落的映射，以便找到前一段和后一段
        all_paragraphs = {}  # (section_number, para_index) -> para_content
        
        # 第一轮：收集所有段落
        for obj_str in json_objects:
            obj_str = obj_str.strip()
            if not obj_str or obj_str.startswith('//'):
                continue
            
            try:
                data = json.loads(obj_str)
                section_number = data['section']['number']
                
                for para in data.get('paragraphs', []):
                    para_index = para['index']
                    all_paragraphs[(section_number, para_index)] = para['content']
            except json.JSONDecodeError:
                print(f"Invalid JSON object: {obj_str[:100]}...")
                continue
        
        # 第二轮：提取包含关键词的句子
        for obj_str in json_objects:
            obj_str = obj_str.strip()
            if not obj_str or obj_str.startswith('//'):
                continue
            
            try:
                data = json.loads(obj_str)
                if 'paragraphs' not in data:
                    continue
                
                section = data['section']
                section_number = section['number']
                
                for para in data['paragraphs']:
                    para_index = para['index']
                    para_content = para['content']
                    
                    # 获取前一段和后一段
                    prev_para = all_paragraphs.get((section_number, para_index - 1), "")
                    next_para = all_paragraphs.get((section_number, para_index + 1), "")
                    
                    if not para.get('sentences'):
                        # 如果没有句子, 尝试分割段落内容
                        sentences = []
                        for i, sent in enumerate(self.split_sentences(para_content), 1):
                            sentences.append({
                                'content': sent,
                                'index': i
                            })
                        para['sentences'] = sentences
                    
                    for sent in para['sentences']:
                        sent_content = sent['content']
                        sent_index = sent['index']
                        
                        # 查找关键词
                        found_keywords = []
                        for keyword in self.keywords:
                            pos = sent_content.find(keyword)
                            if pos >= 0:
                                # 确保找到的是独立的关键词, 而不是更长词汇的一部分
                                after_word = sent_content[pos:pos+len(keyword)+4].strip()
                                if after_word == keyword or not after_word.startswith(keyword + " NOT"):
                                    found_keywords.append(keyword)
                        
                        # 如果找到关键词，添加到结果
                        if found_keywords:
                            cleaned_content = self.clean_sentence(sent_content)
                            result = {
                                "sr": cleaned_content,
                                "keywords": found_keywords,
                                "section": section,
                                "para_index": para_index,
                                "sentence_index": sent_index,
                                "current_para": para_content,
                                "prev_para": prev_para,
                                "next_para": next_para
                            }
                            results.append(result)
            except json.JSONDecodeError:
                print(f"Invalid JSON object: {obj_str[:100]}...")
                continue
        
        # 保存JSONL结果
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for result in results:
                # 创建一个适合JSONL格式的副本
                jsonl_entry = {
                    "sr": result["sr"],
                    "keywords": result["keywords"],
                    "section": result["section"],
                    "para_index": result["para_index"],
                    "sentence_index": result["sentence_index"]
                }
                f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
        
        # 保存CSV结果
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            # 写入CSV头
            csv_writer.writerow([
                "SR Text", "Keywords", "Current Paragraph", "Previous Paragraph", 
                "Next Paragraph", "Section Number", "Section Title", "Section Level", 
                "Paragraph Index", "Sentence Index"
            ])
            
            # 写入每一行数据
            for result in results:
                keywords_str = ", ".join(result["keywords"])
                
                # 清理段落文本以便CSV展示
                current_para = self.clean_sentence(result["current_para"])
                prev_para = self.clean_sentence(result["prev_para"])
                next_para = self.clean_sentence(result["next_para"])
                
                csv_writer.writerow([
                    result["sr"],
                    keywords_str,
                    current_para,
                    prev_para,
                    next_para,
                    result["section"]["number"],
                    result["section"]["title"],
                    result["section"]["level"],
                    result["para_index"],
                    result["sentence_index"]
                ])
        
        print(f"已提取 {len(results)} 个包含关键词的句子")
        print(f"JSONL结果已保存到: {output_jsonl_path}")
        print(f"CSV结果已保存到: {output_csv_path}")

def process_rfc_file(rfc_id, output_base_dir):
    """处理RFC文件，执行解析和SR提取
    
    Args:
        input_txt_path: 输入的RFC文本文件路径
        output_base_dir: 输出目录基路径
    """
    # 创建输出目录
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_txt_path = Path("../raw_rfc") / f"rfc{rfc_id}.txt"
    
    # 设置输出文件路径
    jsonl_path = output_dir / f"{rfc_id}para.jsonl"
    pretty_jsonl_path = output_dir / f"{rfc_id}para_pretty.jsonl"
    sr_jsonl_path = output_dir / f"{rfc_id}sr.jsonl"
    sr_csv_path = output_dir / f"{rfc_id}sr.csv"
    
    # 创建解析器
    parser = RFCParser()
    
    # 读取输入文件
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析文档
    print(f"解析文档: {input_txt_path}")
    doc_structure = parser.parse_document(content)
    
    # 保存JSONL格式
    # print(f"保存JSONL格式到: {jsonl_path}")
    # parser.save_to_jsonl(doc_structure, jsonl_path, pretty=False)
    
    # 保存美化后的JSONL格式
    print(f"保存美化的JSONL格式到: {pretty_jsonl_path}")
    parser.save_to_jsonl(doc_structure, pretty_jsonl_path, pretty=True)
    
    # 提取包含关键词的句子
    print("提取包含关键词的句子...")
    parser.extract_sentences_with_keywords(pretty_jsonl_path, sr_jsonl_path, sr_csv_path)
    
    print("处理完成!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RFC解析器和SR提取器')
    parser.add_argument('input_file', help='输入的RFC文本文件路径')
    parser.add_argument('--output-dir', '-o', default='../sr', help='输出目录路径')
    
    args = parser.parse_args()
    
    process_rfc_file(args.input_file, args.output_dir)