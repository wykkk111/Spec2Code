import asyncio
import sys
from datetime import datetime
import pandas as pd
import uvloop
import os
import json
import re
from volcenginesdkarkruntime import AsyncArk
from openai import AsyncOpenAI
import argparse
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI


class ProtocolFilterResponse(BaseModel):
    relevance: bool
    reason: str

def parse_args():
    parser = argparse.ArgumentParser(description='Process functions with different models and protocols')
    parser.add_argument('-m', type=str, choices=['pro', 'lite', 'dsv3', 'deepseek', 'openai'], 
                       required=True, help='Model name to use (pro, lite, dsv3, deepseek, or openai)')
    parser.add_argument('-p', type=str, choices=['http', 'tls'], 
                       required=True, help='Protocol to filter (http or tls)')
    parser.add_argument('-w', type=int, default=50,
                       help='Number of concurrent workers (default: 50)')
    return parser.parse_args()

def load_prompts(protocol):
    """Load system and user prompts for the specified protocol."""
    prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompt')
    
    with open(os.path.join(prompt_dir, f'{protocol}_prompt.txt'), 'r') as f:
        sys_prompt = f.read()
    
    with open(os.path.join(prompt_dir, f'{protocol}_user_prompt.txt'), 'r') as f:
        user_prompt = f.read()
    
    return sys_prompt, user_prompt

async def process_batch_volcengine(batch_df, worker_id, client, sys_prompt, user_prompt, endpoint):
    results = []
    for idx, row in batch_df.iterrows():
        print(f"Worker {worker_id} processing function ID: {row['function_id']}")
        try:
            completion = await client.chat.completions.create(
                model=endpoint,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt + row['source_code']},
                ],
                max_tokens=400,
                temperature=0
            )
            res = completion.choices[0].message.content
            print(f"Worker {worker_id} received response: {res[:100]}...")
            
            try:
                # Extract JSON content if it's wrapped in markdown code blocks
                json_content = extract_json_from_response(res)
                
                data = json.loads(json_content)
                data_lower = {k.lower(): v for k, v in data.items()}

                relevance = data_lower.get("relevance")
                reason = data_lower.get("reason")
                result = {
                    'function_id': row['function_id'],
                    'function_name': row['function_name'],
                    'Relevance': str(relevance).lower(),
                    'Reason': reason,
                    'status': 'success',
                    'api_response': None
                }
            except json.JSONDecodeError as json_err:
                print(f"Worker {worker_id} JSON parsing error for function ID {row['function_id']}: {json_err}")
                print(f"Raw response: {res}")
                result = {
                    'function_id': row['function_id'],
                    'function_name': row['function_name'],
                    'Relevance': None,
                    'Reason': None,
                    'status': 'failed',
                    'api_response': f"JSON parsing error: {res[:500]}"
                }
        except Exception as e:
            print(f"Worker {worker_id} failed processing function ID {row['function_id']}: {e}")
            result = {
                'function_id': row['function_id'],
                'function_name': row['function_name'],
                'Relevance': None,
                'Reason': None,
                'status': 'failed',
                'api_response': str(e)
            }
        results.append(result)
        print(f"Worker {worker_id} completed function ID: {row['function_id']}")
    return results

async def process_batch_deepseek(batch_df, worker_id, client, sys_prompt, user_prompt):
    results = []
    for idx, row in batch_df.iterrows():
        print(f"Worker {worker_id} processing function ID: {row['function_id']}")
        try:
            completion = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt + row['source_code']},
                ],
                max_tokens=400,
                temperature=0
            )
            res = completion.choices[0].message.content
            print(f"Worker {worker_id} received response: {res[:100]}...")
            
            try:
                # Extract JSON content if it's wrapped in markdown code blocks
                json_content = extract_json_from_response(res)
                
                data = json.loads(json_content)
                data_lower = {k.lower(): v for k, v in data.items()}

                relevance = data_lower.get("relevance")
                reason = data_lower.get("reason")
                result = {
                    'function_id': row['function_id'],
                    'function_name': row['function_name'],
                    'Relevance': str(relevance).lower(),
                    'Reason': reason,
                    'status': 'success',
                    'api_response': None
                }
            except json.JSONDecodeError as json_err:
                print(f"Worker {worker_id} JSON parsing error for function ID {row['function_id']}: {json_err}")
                print(f"Raw response: {res}")
                result = {
                    'function_id': row['function_id'],
                    'function_name': row['function_name'],
                    'Relevance': None,
                    'Reason': None,
                    'status': 'failed',
                    'api_response': f"JSON parsing error: {res[:500]}"
                }
        except Exception as e:
            print(f"Worker {worker_id} failed processing function ID {row['function_id']}: {e}")
            result = {
                'function_id': row['function_id'],
                'function_name': row['function_name'],
                'Relevance': None,
                'Reason': None,
                'status': 'failed',
                'api_response': str(e)
            }
        results.append(result)
        print(f"Worker {worker_id} completed function ID: {row['function_id']}")
    return results

async def process_batch_openai(batch_df, worker_id, client, sys_prompt, user_prompt):
    results = []
    for idx, row in batch_df.iterrows():
        print(f"Worker {worker_id} processing function ID: {row['function_id']}")
        try:
            completion = await client.beta.chat.completions.parse(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt + row['source_code']},
                ],
                # max_tokens=400,
                # temperature=0,
                response_format=ProtocolFilterResponse
            )
            res = completion.choices[0].message.parsed
            print(f"Worker {worker_id} received response: relevance={res.relevance}, reason={res.reason[:100]}...")
            result = {
                'function_id': row['function_id'],
                'function_name': row['function_name'],
                'Relevance': res.relevance,
                'Reason': res.reason,
                'status': 'success',
                'api_response': None
            }
        except Exception as e:
            print(f"Worker {worker_id} failed processing function ID {row['function_id']}: {e}")
            result = {
                'function_id': row['function_id'],
                'function_name': row['function_name'],
                'Relevance': None,
                'Reason': None,
                'status': 'failed',
                'api_response': str(e)
            }
        results.append(result)
        print(f"Worker {worker_id} completed function ID: {row['function_id']}")
    return results

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

async def worker(worker_id, df_chunk, sys_prompt, user_prompt, args):
    if args.m == 'deepseek':
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        results = await process_batch_deepseek(df_chunk, worker_id, client, sys_prompt, user_prompt)
    elif args.m == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = AsyncOpenAI(api_key=api_key)
        results = await process_batch_openai(df_chunk, worker_id, client, sys_prompt, user_prompt)
    else:
        client = AsyncArk(api_key="xxx")
        endpoint = get_endpoint(args.m)
        results = await process_batch_volcengine(df_chunk, worker_id, client, sys_prompt, user_prompt, endpoint)
    
    print(f"Worker {worker_id} completed processing.")
    return results

def get_endpoint(model_name):
    """Get the endpoint for the specified model."""
    if model_name == "pro":
        return "ep-20250213145836-rx645"
    elif model_name == "lite":
        return "ep-20250213150054-sgjvj"
    elif model_name == "dsv3":
        return "ep-20250211123406-cngqn"
        # return "ep-20250210193928-fl84t" # ds r1 dstill
    else:
        raise ValueError(f"Invalid model name: {model_name}")

async def main():
    args = parse_args()
    
    # Load prompts
    sys_prompt, user_prompt = load_prompts(args.p)
    
    # Set input and output paths based on protocol
    if args.p == "http":
        input_csv = '/data/a/ykw/RFC/final/data/httpd/func/all_func.csv'
        output_csv = f"../data/httpd/filter1_{args.m}r1.csv"
        relevant_output_csv = f"../data/httpd/filter1_{args.m}r1_relevant.csv"
    else:  # tls
        input_csv = '/data/a/ykw/RFC/final/data/boringssl/func/all_func.csv'
        output_csv = f"../data/boringssl/filter1_{args.m}.csv"
        relevant_output_csv = f"../data/boringssl/filter1_{args.m}_relevant.csv"
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    # df = df.head(20)
    print(f"Processing all {len(df)} functions from {input_csv}")
    
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
        tasks.append(worker(i, df_chunk, sys_prompt, user_prompt, args))

    # Wait for all tasks to complete and collect results
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results list
    final_results = [item for sublist in all_results for item in sublist]
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(output_csv, index=False)
    
    # 在合并前删除results_df中的function_name列
    if 'function_name' in results_df.columns:
        results_df = results_df.drop(columns=['function_name'])

    # 然后正常合并，不需要后缀
    merged_df = pd.merge(df, results_df, on='function_id', how='left')
    
    # Filter for non-false relevance (handle both string and boolean values)
    if 'Relevance' in merged_df.columns:
        # Check the type of the first non-null value to determine how to filter
        non_null_values = merged_df['Relevance'].dropna()
        if len(non_null_values) > 0:
            first_value = non_null_values.iloc[0]
            if isinstance(first_value, bool):
                # Handle boolean values
                relevant_df = merged_df[merged_df['Relevance'] != False]
            else:
                # Handle string values
                relevant_df = merged_df[merged_df['Relevance'].astype(str).str.lower() != 'false']
        else:
            # If all values are null, consider all as relevant
            relevant_df = merged_df
    else:
        relevant_df = pd.DataFrame()
    
    # If there are any relevant results, save them
    if not relevant_df.empty:
        # Save the relevant functions to a new CSV
        relevant_df.to_csv(relevant_output_csv, index=False)
        print(f"Found {len(relevant_df)} relevant functions, saved to {relevant_output_csv}")
    else:
        print("No relevant functions found")
    
    end = datetime.now()
    print(f"Total time: {end - start}")
    print(f"Total functions processed: {len(final_results)}")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(main())
    else:
        uvloop.install()
        asyncio.run(main()) 