import pandas as pd
import os

def extract_fully_matches(protocol):
    # 构建文件路径
    file_path = f'/data/a/ykw/RFC/final/data/{protocol}/final_verification_results.csv'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {protocol}: {file_path}")
        return None
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
        # 筛选出outcome为"Fully Matches"的行
        fully_matches = df[df['outcome'] == 'Fully Matches']
        return fully_matches
    except Exception as e:
        print(f"Error processing {protocol}: {str(e)}")
        return None

def main():
    # 所有需要处理的协议
    protocols = ['boringssl', 'httpd', 'nginx', 'openssl']
    
    # 存储所有结果
    all_results = {}
    
    # 处理每个协议
    for protocol in protocols:
        results = extract_fully_matches(protocol)
        if results is not None:
            all_results[protocol] = results
            print(f"\nResults for {protocol}:")
            print(f"Found {len(results)} fully matching entries")
            print(results)
            
            # 可选：保存到新的CSV文件
            output_path = f'/data/a/ykw/RFC/final/data/{protocol}/fully_matches.csv'
            results.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
