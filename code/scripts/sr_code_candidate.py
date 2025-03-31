import pandas as pd
import os
import re

def clean_keyword(keyword):
    """清理关键词，移除JSON格式符号如方括号和引号"""
    # 移除 [ " ] 等JSON格式字符
    cleaned = re.sub(r'[\[\]"]', '', keyword.strip())
    return cleaned

def process_files():
    # Define file paths
    dual_filter_path = '/data/a/ykw/RFC/final/data/httpd/dual_filter_results.csv'
    msm_filter_path = '/data/a/ykw/RFC/final/sr/msm_filtered.csv'
    
    # Read CSV files
    dual_df = pd.read_csv(dual_filter_path)
    msm_df = pd.read_csv(msm_filter_path)
    
    # Print column names to verify
    print("Columns in msm_filtered.csv:", msm_df.columns.tolist())
    print("Columns in dual_filter_results.csv:", dual_df.columns.tolist())
    
    # Add index as column to msm_df to track the original SR IDs
    msm_df['sr_id'] = msm_df.index
    
    # Define columns to check in dual_filter_results.csv
    columns_to_check = ['status_code', 'field_name', 'content_coding', 'method', 'syntax', 'matched_subcategories']
    
    # Define column mapping between dual_filter_results.csv and msm_filtered.csv
    column_mapping = {
        'status_code': 'status_code',
        'field_name': 'field_name',
        'content_coding': 'content_coding', 
        'method': 'method',
        'syntax': 'syntax',
        'matched_subcategories': 'class'  # Map 'matched_subcategories' to 'class'
    }
    
    # Check if all columns exist in msm_df
    for msm_col in column_mapping.values():
        if msm_col not in msm_df.columns:
            print(f"Warning: Column '{msm_col}' not found in msm_filtered.csv")
    
    # Create a new column for SR IDs
    dual_df['matching_sr_ids'] = None
    
    # Keep track of matching statistics
    total_rows = len(dual_df)
    rows_with_matches = 0
    total_keywords = 0
    matched_keywords = 0
    
    # 记录所有匹配的SR ID
    all_matched_sr_ids = []
    
    # Process each row in dual_filter_results.csv
    for idx, row in dual_df.iterrows():
        matching_sr_ids = []
        row_has_match = False
        
        # Skip rows where all specified columns are empty
        if all(pd.isna(row[col]) or row[col] == '' for col in columns_to_check):
            dual_df.drop(idx, inplace=True)
            continue
        
        # Process each column
        for col in columns_to_check:
            if pd.notna(row[col]) and row[col] != '':
                # Split multiple keywords in the column
                raw_keywords = str(row[col]).split(',')
                
                for raw_keyword in raw_keywords:
                    # 清理关键词，移除JSON格式符号
                    keyword = clean_keyword(raw_keyword)
                    
                    if keyword:
                        total_keywords += 1
                        # Use the mapped column name for msm_filtered.csv
                        msm_col = column_mapping[col]
                        
                        if msm_col not in msm_df.columns:
                            continue
                        
                        # Print sample data to check
                        if idx < 5:  # Only print for first few rows to avoid overload
                            print(f"Searching for keyword '{keyword}' (from '{raw_keyword}') in column '{msm_col}'")
                            # Show some sample values from msm_df[msm_col]
                            sample_values = msm_df[msm_col].dropna().sample(min(3, len(msm_df[msm_col].dropna()))).tolist()
                            print(f"Sample values in {msm_col}: {sample_values}")
                        
                        # Find matching rows in msm_filter, treating the keyword as a literal string, not regex
                        matching_rows = msm_df[msm_df[msm_col].notna() & msm_df[msm_col].str.contains(keyword, na=False, regex=False)]
                        
                        if not matching_rows.empty:
                            matched_keywords += 1
                            row_has_match = True
                            # Add SR IDs to the list
                            sr_ids = matching_rows['sr_id'].tolist()
                            matching_sr_ids.extend(sr_ids)
                            all_matched_sr_ids.extend(sr_ids)  # 添加到总列表中
                            
                            if idx < 5:  # Only print for first few rows
                                print(f"Found {len(sr_ids)} matches for '{keyword}' in column '{msm_col}'")
        
        # Remove duplicates and store the list
        matching_sr_ids = list(set(matching_sr_ids))
        dual_df.at[idx, 'matching_sr_ids'] = ','.join(map(str, matching_sr_ids)) if matching_sr_ids else None
        
        if row_has_match:
            rows_with_matches += 1
    
    # 删除那些没有匹配到任何SR的函数条目
    initial_rows = len(dual_df)
    dual_df = dual_df.dropna(subset=['matching_sr_ids'])
    final_rows = len(dual_df)
    
    print(f"\n已删除 {initial_rows - final_rows} 个没有匹配SR的函数条目")
    
    # 计算总的匹配SR ID数量（包括重复的）
    total_matched_sr_ids = len(all_matched_sr_ids)
    # 计算不同的SR ID数量（去重后的）
    unique_matched_sr_ids = len(set(all_matched_sr_ids))
    
    # Print matching statistics
    print(f"\nMatching Statistics:")
    print(f"Total rows processed: {total_rows}")
    print(f"Rows with at least one match: {rows_with_matches}")
    print(f"Rows retained after filtering: {final_rows}")
    print(f"Total keywords checked: {total_keywords}")
    print(f"Keywords that matched: {matched_keywords}")
    print(f"Total matched SR IDs (including duplicates): {total_matched_sr_ids}")
    print(f"Unique matched SR IDs: {unique_matched_sr_ids}")
    
    # Save the result
    output_path = os.path.join(os.path.dirname(dual_filter_path), 'dual_filter_with_sr_ids.csv')
    dual_df.to_csv(output_path, index=False)
    print(f"\nProcessed file saved to: {output_path}")

if __name__ == "__main__":
    process_files()
