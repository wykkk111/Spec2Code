import asyncio
import sys
import os
import json
import re
import pandas as pd
from datetime import datetime
import uvloop
from volcenginesdkarkruntime import AsyncArk
import argparse
from typing import List, Dict, Any

def load_prompts():
    """Load system prompts for both TLS filters."""
    prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompt')
    
    with open(os.path.join(prompt_dir, 'tls_kw.txt'), 'r') as f:
        kw_prompt = f.read()
    
    with open(os.path.join(prompt_dir, 'tls_nonkw.txt'), 'r') as f:
        nonkw_prompt = f.read()
    
    return kw_prompt, nonkw_prompt

def extract_json_from_response(response):
    """
    Extract JSON content from a response that might be wrapped in markdown code blocks.
    """
    # Check if the response is wrapped in markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response)
    
    if match:
        # Return the content inside the code blocks
        return match.group(1).strip()
    
    # If no code blocks found, return the original response
    return response

def get_endpoint(model_name):
    """Get the endpoint for the specified model."""
    if model_name == "pro":
        return "ep-20250213145836-rx645"
    elif model_name == "lite":
        return "ep-20250213150054-sgjvj"
    elif model_name == "dsv3":
        return "ep-20250211123406-cngqn"
    elif model_name == "dsr1":
        return "ep-20250210143928-8945x"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

async def process_function_with_volcengine(function_id, function_name, source_code, client, endpoint, kw_prompt, nonkw_prompt, worker_id):
    """Process a single function with both TLS prompts using Volcano Engine API."""
    print(f"Worker {worker_id} processing function ID: {function_id}")
    result = {
        'function_id': function_id,
        'function_name': function_name,
        'ProtocolCoreVersion': [],
        'HandshakeFlow': [],
        'KeyExchangeSecurity': [],
        'AuthenticationCertificates': [],
        'DataTransferRecordLayer': [],
        'SessionManagementAlerts': [],
        'matched_subcategories': []
    }
    
    try:
        # Process with TLS keyword filter prompt
        kw_completion = await client.chat.completions.create(
            model=endpoint,
            messages=[
                {"role": "system", "content": kw_prompt},
                {"role": "user", "content": f"```\n{source_code}\n```"}
            ],
            max_tokens=1000,
            temperature=0
        )
        kw_res = kw_completion.choices[0].message.content
        print(f"Worker {worker_id} received TLS keyword response: {kw_res[:100]}...")
        
        # Process with TLS non-keyword filter prompt
        nonkw_completion = await client.chat.completions.create(
            model="ep-20250210193928-fl84t", #r1dstill
            messages=[
                {"role": "system", "content": nonkw_prompt},
                {"role": "user", "content": f"```\n{source_code}\n```"}
            ],
            max_tokens=1000,
            temperature=0
        )
        nonkw_res = nonkw_completion.choices[0].message.content
        print(f"Worker {worker_id} received TLS classification response: {nonkw_res[:100]}...")
        
        # Extract JSON content from both responses
        kw_json = extract_json_from_response(kw_res)
        nonkw_json = extract_json_from_response(nonkw_res)
        
        try:
            # Parse TLS keyword response
            kw_data = json.loads(kw_json)
            result['ProtocolCoreVersion'] = kw_data.get('ProtocolCoreVersion', [])
            result['HandshakeFlow'] = kw_data.get('HandshakeFlow', [])
            result['KeyExchangeSecurity'] = kw_data.get('KeyExchangeSecurity', [])
            result['AuthenticationCertificates'] = kw_data.get('AuthenticationCertificates', [])
            result['DataTransferRecordLayer'] = kw_data.get('DataTransferRecordLayer', [])
            result['SessionManagementAlerts'] = kw_data.get('SessionManagementAlerts', [])
            
            # Parse TLS classification response
            nonkw_data = json.loads(nonkw_json)
            result['matched_subcategories'] = nonkw_data.get('matched_subcategories', [])
            
            result['status'] = 'success'
            result['error'] = None
            
        except json.JSONDecodeError as json_err:
            print(f"Worker {worker_id} JSON parsing error for function ID {function_id}: {json_err}")
            print(f"TLS Keyword response raw: {kw_res[:500]}")
            print(f"TLS Classification response raw: {nonkw_res[:500]}")
            result['status'] = 'failed'
            result['error'] = f"JSON parsing error: {str(json_err)}"
            
    except Exception as e:
        print(f"Worker {worker_id} failed processing function ID {function_id}: {e}")
        result['status'] = 'failed'
        result['error'] = str(e)
    
    print(f"Worker {worker_id} completed function ID: {function_id}")
    return result

async def worker(worker_id, df_chunk, model_name, kw_prompt, nonkw_prompt):
    """Worker that processes a chunk of functions."""
    client = AsyncArk(api_key="xxxx")
    endpoint = get_endpoint(model_name)
    results = []
    
    for idx, row in df_chunk.iterrows():
        result = await process_function_with_volcengine(
            row['function_id'], 
            row['function_name'], 
            row['source_code'], 
            client, 
            endpoint,
            kw_prompt, 
            nonkw_prompt, 
            worker_id
        )
        results.append(result)
    
    print(f"Worker {worker_id} completed processing {len(df_chunk)} functions.")
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process functions with TLS keyword and classification filters')
    parser.add_argument('-m', type=str, choices=['pro', 'lite', 'dsv3'], 
                       required=True, help='Model name to use (pro, lite, or dsv3)')
    parser.add_argument('-w', type=int, default=20,
                       help='Number of concurrent workers (default: 20)')
    parser.add_argument('-i', type=str, default='../data/boringssl/filter1_dsv3_relevant.csv',
                       help='Input CSV file path (default: ../data/boringssl/filter1_dsv3_relevant.csv)')
    parser.add_argument('-o', type=str, default='/data/a/ykw/RFC/final/data/boringssl/tls_dual_filter_results.csv',
                       help='Output CSV file path (default: ../data/boringssl/tls_dual_filter_results.csv)')
    return parser.parse_args()

async def main():
    """Main function to run the TLS dual filter processing."""
    args = parse_args()
    
    # Load TLS prompts
    kw_prompt, nonkw_prompt = load_prompts()
    
    # Read the CSV file
    df = pd.read_csv(args.i)
    # df = df.head(5)
    print(f"Processing {len(df)} functions from {args.i}")
    
    # Configure batch processing
    max_concurrent_tasks = args.w
    chunk_size = len(df) // max_concurrent_tasks + (1 if len(df) % max_concurrent_tasks else 0)
    
    start = datetime.now()
    
    # Split DataFrame into chunks and create tasks
    tasks = []
    for i in range(max_concurrent_tasks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        if start_idx >= len(df):
            break
        df_chunk = df.iloc[start_idx:end_idx]
        tasks.append(worker(i, df_chunk, args.m, kw_prompt, nonkw_prompt))

    # Wait for all tasks to complete and collect results
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results list
    final_results = [item for sublist in all_results for item in sublist]
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(final_results)
    
    # Convert list columns to strings for easier CSV handling
    for col in ['ProtocolCoreVersion', 'HandshakeFlow', 'KeyExchangeSecurity', 
               'AuthenticationCertificates', 'DataTransferRecordLayer', 
               'SessionManagementAlerts', 'matched_subcategories']:
        results_df[col] = results_df[col].apply(lambda x: json.dumps(x))
    
    # Drop function_name to avoid duplication when merging
    if 'function_name' in results_df.columns:
        results_df = results_df.drop(columns=['function_name'])
    
    # Merge with original data on function_id
    merged_df = pd.merge(df, results_df, on='function_id', how='left')
    
    # Save the results to a new CSV
    merged_df.to_csv(args.o, index=False)
    
    end = datetime.now()
    print(f"Total time: {end - start}")
    print(f"Total functions processed: {len(final_results)}")
    print(f"Results saved to {args.o}")

if __name__ == "__main__":
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(main())
    else:
        uvloop.install()
        asyncio.run(main())
