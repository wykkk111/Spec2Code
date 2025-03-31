import pandas as pd
import os

def analyze_http_protocol():
    print("\n=== HTTP Protocol Analysis ===")
    # 读取原始CSV文件
    df = pd.read_csv('/data/a/ykw/RFC/final/sr/msm_filtered.csv')

    # 选择需要的列并创建新的DataFrame
    selected_columns = ['First_Keyword', 'status_code', 'field_name', 
                       'content_coding', 'method', 'syntax']
    new_df = df[selected_columns].copy()

    # 添加httpd和nginx列，初始值设为0
    new_df['httpd'] = 0
    new_df['nginx'] = 0

    # 定义处理CSV文件的函数
    def process_verification_csv(csv_path, implementation):
        verification_df = pd.read_csv(csv_path)
        
        # 过滤出Fully Matches和Partially Matches的结果
        matched_df = verification_df[
            (verification_df['outcome'] == 'Fully Matches') | 
            (verification_df['outcome'] == 'Partially Matches')
        ]
        
        # 统计每个sr_index的出现次数
        sr_counts = matched_df['sr_index'].value_counts()
        
        # 更新DataFrame中的计数
        for sr_index, count in sr_counts.items():
            if sr_index < len(new_df):
                new_df.at[sr_index, implementation] = count

    # 处理指定的CSV文件
    httpd_csv = '/data/a/ykw/RFC/final/data/httpd/final_verification_results.csv'
    nginx_csv = '/data/a/ykw/RFC/final/data/nginx/final_verification_results.csv'

    # 处理httpd和nginx的CSV文件
    process_verification_csv(httpd_csv, 'httpd')
    process_verification_csv(nginx_csv, 'nginx')

    # 定义类别和关键词
    categories = ['status_code', 'field_name', 'content_coding', 'method', 'syntax']
    keywords = ['MUST', 'SHOULD', 'MAY']
    implementations = ['httpd', 'nginx']

    # 打印表格头部
    print("\n{:<12}{:<12}{:<15}{:<15}{:<15}{:<15}{:<15}".format(
        "Impl", "Keyword", "Status", "Field", "Content", "Method", "Syntax"
    ))
    print("-" * 87)

    # 生成表格内容
    for impl in implementations:
        for keyword in keywords:
            row = [impl.capitalize(), keyword]
            
            # 计算每个类别的覆盖数量和总数
            for category in categories:
                # 获取该类别不为空且不为0且关键词匹配的行
                category_df = new_df[
                    (new_df[category].notna()) & 
                    (new_df[category] != 0) & 
                    (new_df['First_Keyword'] == keyword)
                ]
                total = len(category_df)
                covered = len(category_df[category_df[impl] > 0])
                row.append(f"{covered}/{total}")
            
            # 打印行
            print("{:<12}{:<12}{:<15}{:<15}{:<15}{:<15}{:<15}".format(*row))
        
        # 在每个实现后添加空行
        print()

    # 保存结果到新的CSV文件
    new_df.to_csv('http_verification_implementation_counts.csv', index=False)

    # 分析不同关键词的覆盖情况
    print("\nCoverage Analysis by First_Keyword:")
    for keyword in keywords:
        keyword_df = new_df[new_df['First_Keyword'] == keyword]
        total = len(keyword_df)
        httpd_covered = len(keyword_df[keyword_df['httpd'] > 0])
        nginx_covered = len(keyword_df[keyword_df['nginx'] > 0])
        
        print(f"\n{keyword}:")
        print(f"Total requirements: {total}")
        print(f"HTTPD coverage: {httpd_covered} ({httpd_covered/total*100:.2f}%)")
        print(f"Nginx coverage: {nginx_covered} ({nginx_covered/total*100:.2f}%)")

    # 分析实现之间的覆盖情况
    print("\nImplementation Coverage Analysis:")
    total = len(new_df)
    both_covered = len(new_df[(new_df['httpd'] > 0) & (new_df['nginx'] > 0)])
    only_httpd = len(new_df[(new_df['httpd'] > 0) & (new_df['nginx'] == 0)])
    only_nginx = len(new_df[(new_df['httpd'] == 0) & (new_df['nginx'] > 0)])
    neither_covered = len(new_df[(new_df['httpd'] == 0) & (new_df['nginx'] == 0)])

    print(f"Total requirements: {total}")
    print(f"Both implementations cover: {both_covered} ({both_covered/total*100:.2f}%)")
    print(f"Only HTTPD covers: {only_httpd} ({only_httpd/total*100:.2f}%)")
    print(f"Only Nginx covers: {only_nginx} ({only_nginx/total*100:.2f}%)")
    print(f"Neither implementation covers: {neither_covered} ({neither_covered/total*100:.2f}%)")

    # 按关键词分析实现之间的覆盖情况
    print("\nImplementation Coverage Analysis by First_Keyword:")
    for keyword in keywords:
        keyword_df = new_df[new_df['First_Keyword'] == keyword]
        total = len(keyword_df)
        both_covered = len(keyword_df[(keyword_df['httpd'] > 0) & (keyword_df['nginx'] > 0)])
        only_httpd = len(keyword_df[(keyword_df['httpd'] > 0) & (keyword_df['nginx'] == 0)])
        only_nginx = len(keyword_df[(keyword_df['httpd'] == 0) & (keyword_df['nginx'] > 0)])
        neither_covered = len(keyword_df[(keyword_df['httpd'] == 0) & (keyword_df['nginx'] == 0)])
        
        print(f"\n{keyword}:")
        print(f"Total requirements: {total}")
        print(f"Both implementations cover: {both_covered} ({both_covered/total*100:.2f}%)")
        print(f"Only HTTPD covers: {only_httpd} ({only_httpd/total*100:.2f}%)")
        print(f"Only Nginx covers: {only_nginx} ({only_nginx/total*100:.2f}%)")
        print(f"Neither implementation covers: {neither_covered} ({neither_covered/total*100:.2f}%)")

    # 添加HTTP特定类别分析
    print("\nCoverage Analysis by HTTP Category:")
    http_categories = ['status_code', 'field_name', 'method', 'syntax']
    
    for category in http_categories:
        category_df = new_df[new_df[category] == 1]
        if len(category_df) > 0:
            total = len(category_df)
            httpd_covered = len(category_df[category_df['httpd'] > 0])
            nginx_covered = len(category_df[category_df['nginx'] > 0])
            
            # 计算覆盖情况
            both_covered = len(category_df[(category_df['httpd'] > 0) & (category_df['nginx'] > 0)])
            only_httpd = len(category_df[(category_df['httpd'] > 0) & (category_df['nginx'] == 0)])
            only_nginx = len(category_df[(category_df['httpd'] == 0) & (category_df['nginx'] > 0)])
            neither_covered = len(category_df[(category_df['httpd'] == 0) & (category_df['nginx'] == 0)])
            
            print(f"\n{category}:")
            print(f"Total requirements: {total}")
            print(f"HTTPD coverage: {httpd_covered} ({httpd_covered/total*100:.2f}%)")
            print(f"Nginx coverage: {nginx_covered} ({nginx_covered/total*100:.2f}%)")
            print(f"Both implementations cover: {both_covered} ({both_covered/total*100:.2f}%)")
            print(f"Only HTTPD covers: {only_httpd} ({only_httpd/total*100:.2f}%)")
            print(f"Only Nginx covers: {only_nginx} ({only_nginx/total*100:.2f}%)")
            print(f"Neither implementation covers: {neither_covered} ({neither_covered/total*100:.2f}%)")
            
            # 按关键词分析该类别
            print("\nBreakdown by Keywords:")
            for keyword in keywords:
                keyword_category_df = category_df[category_df['First_Keyword'] == keyword]
                if len(keyword_category_df) > 0:
                    k_total = len(keyword_category_df)
                    k_httpd = len(keyword_category_df[keyword_category_df['httpd'] > 0])
                    k_nginx = len(keyword_category_df[keyword_category_df['nginx'] > 0])
                    
                    print(f"\n{keyword} in {category}:")
                    print(f"Total: {k_total}")
                    print(f"HTTPD: {k_httpd} ({k_httpd/k_total*100:.2f}%)")
                    print(f"Nginx: {k_nginx} ({k_nginx/k_total*100:.2f}%)")

def analyze_tls_protocol():
    print("\n=== TLS Protocol Analysis ===")
    # 读取原始CSV文件
    tls_csv_path = '/data/a/ykw/RFC/final/sr/tls/8446sr_with_keywords.csv'
    
    if not os.path.exists(tls_csv_path):
        print(f"ERROR: TLS CSV file not found at {tls_csv_path}")
        return
        
    df = pd.read_csv(tls_csv_path)

    # From Keywords column extract First_Keyword
    def extract_first_keyword(keywords):
        if pd.isna(keywords):
            return None
        # 按空格或逗号分割，取第一个词
        if isinstance(keywords, str):
            words = keywords.split()
            if words:
                first_word = words[0].strip(',')
                if first_word in ['MUST', 'SHOULD', 'MAY']:
                    return first_word
        return None

    # 添加First_Keyword列
    df['First_Keyword'] = df['Keywords'].apply(extract_first_keyword)
    
    # 确保类别列是数值型的
    category_columns = ['HandshakeFlow', 'KeyExchangeSecurity', 
                       'AuthenticationCertificates', 'DataTransferRecordLayer']
    
    # 处理类别列
    for col in category_columns:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in the DataFrame")
            # Create the column with default value 0
            df[col] = 0
        else:
            # 将列表值转换为数值
            def convert_to_numeric(value):
                if pd.isna(value):
                    return 0
                if isinstance(value, str):
                    # 尝试解析字符串形式的列表
                    try:
                        # 移除方括号和引号，分割成列表
                        value = value.strip('[]').replace("'", "").replace('"', "")
                        if value:
                            return 1
                        return 0
                    except:
                        return 0
                return 0
            
            df[col] = df[col].apply(convert_to_numeric)

    # 选择需要的列并创建新的DataFrame
    selected_columns = ['First_Keyword',
                       'HandshakeFlow', 'KeyExchangeSecurity', 
                       'AuthenticationCertificates', 'DataTransferRecordLayer']
    
    if 'class' in df.columns:
        selected_columns.append('class')
        
    new_df = df[selected_columns].copy()

    # 添加openssl和boringssl列，初始值设为0
    new_df['openssl'] = 0
    new_df['boringssl'] = 0

    # 定义处理CSV文件的函数
    def process_verification_csv(csv_path, implementation):
        if not os.path.exists(csv_path):
            print(f"ERROR: Verification CSV file not found at {csv_path}")
            return
            
        verification_df = pd.read_csv(csv_path)
        
        if 'outcome' not in verification_df.columns or 'sr_index' not in verification_df.columns:
            print(f"ERROR: Required columns not found in verification CSV. Columns: {verification_df.columns.tolist()}")
            return
        
        # 过滤出Fully Matches和Partially Matches的结果
        matched_df = verification_df[
            (verification_df['outcome'] == 'Fully Matches') | 
            (verification_df['outcome'] == 'Partially Matches')
        ]
        
        # 统计每个sr_index的出现次数
        sr_counts = matched_df['sr_index'].value_counts()
        
        # 更新DataFrame中的计数
        for sr_index, count in sr_counts.items():
            if 0 <= sr_index < len(new_df):  # Make sure sr_index is valid
                new_df.at[sr_index, implementation] = count

    # 处理指定的CSV文件
    openssl_csv = '/data/a/ykw/RFC/final/data/openssl/final_verification_results.csv'
    boringssl_csv = '/data/a/ykw/RFC/final/data/boringssl/final_verification_results.csv'

    # 处理openssl和boringssl的CSV文件
    process_verification_csv(openssl_csv, 'openssl')
    process_verification_csv(boringssl_csv, 'boringssl')

    # 保存结果到新的CSV文件
    new_df.to_csv('tls_verification_implementation_counts.csv', index=False)

    # 定义类别和关键词
    categories = ['HandshakeFlow', 'KeyExchangeSecurity', 
                 'AuthenticationCertificates', 'DataTransferRecordLayer']
    keywords = ['MUST', 'SHOULD', 'MAY']
    implementations = ['openssl', 'boringssl']

    # 打印表格头部
    print("\nImplementation Coverage by Category and Keyword:")
    print("\n{:<15}{:<10}{:<15}{:<15}{:<15}{:<15}".format(
        "Impl", "Keyword", "Handshake", "KeyExchange", "Certificate", "DataTransfer"
    ))
    print("-" * 85)

    # 生成表格内容
    for impl in implementations:
        for keyword in keywords:
            row = [impl.capitalize(), keyword]
            
            # 计算每个类别的覆盖数量和总数
            for category in categories:
                # 获取该类别不为空且不为0且关键词匹配的行
                category_df = new_df[
                    (new_df[category].notna()) & 
                    (new_df[category] != 0) & 
                    (new_df['First_Keyword'] == keyword)
                ]
                total = len(category_df)
                covered = len(category_df[category_df[impl] > 0])
                row.append(f"{covered}/{total}")
            
            # 打印行
            print("{:<15}{:<10}{:<15}{:<15}{:<15}{:<15}".format(*row))
        
        # 在每个实现后添加空行
        print()

if __name__ == "__main__":
    analyze_http_protocol()
    analyze_tls_protocol()
