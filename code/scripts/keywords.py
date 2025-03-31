import json
import re
import pandas as pd

def load_keywords():
    with open('/data/a/ykw/RFC/data/keywords/http.json', 'r') as f:
        return json.load(f)['keywords']

def contains_status_code(text, status_codes):
    matches = []
    
    # Special handling for "100" and "Continue"
    if "100" in status_codes:  # Only check once
        # Match "100" only when it's surrounded by spaces, parentheses or sentence boundaries
        pattern_100 = r'(?:^|\s|\()(100)(?:\s|\)|$)'
        pattern_continue = r'(?:^|\s|\()(Continue)(?:\s|\)|$)'
        
        if re.search(pattern_100, text) or re.search(pattern_continue, text):
            matches.append("100")
            
    # For other status codes
    for code, desc in status_codes.items():
        if code != "100":  # Skip 100 as it's already handled
            if code in text or desc in text:
                matches.append(code)
    
    return matches if matches else ["NONE"]

def contains_field_name(text, fields):
    matches = []
    for field in fields:
        # Case sensitive search using word boundaries
        if re.search(fr'\b{re.escape(field)}\b', text):
            matches.append(field)
    return matches if matches else ["NONE"]

def contains_content_coding(text, codings):
    matches = []
    for coding in codings:
        if coding in text:
            matches.append(coding)
    return matches if matches else ["NONE"]

def contains_method(text, methods):
    matches = []
    for method in methods:
        # Case sensitive search using word boundaries
        if re.search(fr'\b{re.escape(method)}\b', text):
            matches.append(method)
    return matches if matches else ["NONE"]

def contains_syntax(text, syntaxes):
    matches = []
    for syntax in syntaxes:
        # Case sensitive search using word boundaries
        if re.search(fr'\b{re.escape(syntax)}\b', text):
            matches.append(syntax)
    return matches if matches else ["NONE"]

def process_sr(sr_text, keywords):
    terms = {
        "status_code": contains_status_code(sr_text, keywords['status_code']),
        "field": contains_field_name(sr_text, keywords['field_name']),
        "content_coding": contains_content_coding(sr_text, keywords['content_coding']),
        "method": contains_method(sr_text, keywords['method']),
        "syntax": contains_syntax(sr_text, keywords['syntax'])
    }
    return terms

def process_csv_file(input_csv, keywords):
    # Read CSV
    df = pd.read_csv(input_csv)
    
    print("Available columns:", df.columns.tolist())
    
    sr_column = 'SR Text'  # Use the exact column name from the CSV
    
    # Process each SR text
    keyword_results = []
    for sr_text in df[sr_column]:
        terms = process_sr(str(sr_text), keywords)
        keyword_results.append({
            'status_code': ','.join(x for x in terms['status_code'] if x != 'NONE'),
            'field_name': ','.join(x for x in terms['field'] if x != 'NONE'),
            'content_coding': ','.join(x for x in terms['content_coding'] if x != 'NONE'),
            'method': ','.join(x for x in terms['method'] if x != 'NONE'),
            'syntax': ','.join(x for x in terms['syntax'] if x != 'NONE')
        })
    
    # Add new columns
    for key in ['status_code', 'field_name', 'content_coding', 'method', 'syntax']:
        df[key] = [result[key] for result in keyword_results]
    
    return df, keyword_results

def main():
    # Load keywords
    keywords = load_keywords()
    
    RFC_ID = '9110'
    
    input_csv = f'/data/a/ykw/RFC/final/sr/temp_sr/{RFC_ID}sr_constraints.csv'
    output_csv = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_constraints_keywords.csv'
    output_jsonl = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_constraints_keywords.jsonl'
    
    # Process CSV
    df, keyword_results = process_csv_file(input_csv, keywords)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    
    # Create human-readable JSONL
    with open(output_jsonl, 'w') as f:
        for idx, row in df.iterrows():
            readable_item = {
                'id': int(row.get('id', idx + 1)),
                'sr': row['SR Text'],  # Use correct column name
                'keywords_found': {
                    'Status Codes': row['status_code'].split(',') if row['status_code'] else [],
                    'Field Names': row['field_name'].split(',') if row['field_name'] else [],
                    'Content Codings': row['content_coding'].split(',') if row['content_coding'] else [],
                    'Methods': row['method'].split(',') if row['method'] else [],
                    'Syntax': row['syntax'].split(',') if row['syntax'] else []
                }
            }
            f.write(json.dumps(readable_item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
