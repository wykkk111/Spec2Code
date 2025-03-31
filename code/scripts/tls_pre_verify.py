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

def load_prompt():
    """Load system prompt for TLS verification."""
    prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompt')
    
    with open(os.path.join(prompt_dir, 'tls_pre_verify.txt'), 'r') as f:
        verify_prompt = f.read()
    
    return verify_prompt

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

async def verify_function_sr_pair(function_id, function_name, source_code, sr_content, sr_context, sr_index, 
                                 client, endpoint, verify_prompt, worker_id):
    """Verify a single function-SR pair using Volcano Engine API."""
    print(f"Worker {worker_id} processing function ID: {function_id} with SR index: {sr_index}")
    result = {
        'function_id': function_id,
        'function_name': function_name,
        'sr_content': sr_content,
        'sr_index': sr_index,
        'is_match': None,
        'verify_explanation': None
    }
    
    # Include the SR context in the prompt to LLM
    complete_sr = f"SR: {sr_content}\n\nSR Context: {sr_context}"
    
    try:
        # Process with verification prompt
        completion = await client.chat.completions.create(
            model=endpoint,
            messages=[
                {"role": "system", "content": verify_prompt},
                {"role": "user", "content": f"Function code:\n```\n{source_code}\n```\n\n{complete_sr}"}
            ],
            max_tokens=1000,
            temperature=0
        )
        response = completion.choices[0].message.content
        print(f"Worker {worker_id} received verification response: {response[:100]}...")
        
        # Extract JSON content from response
        json_content = extract_json_from_response(response)
        
        try:
            # Parse response
            data = json.loads(json_content)
            result['is_match'] = data.get('is_match', False)
            result['verify_explanation'] = data.get('verify_explanaton', '')
            
            result['status'] = 'success'
            result['error'] = None
            
        except json.JSONDecodeError as json_err:
            print(f"Worker {worker_id} JSON parsing error for function ID {function_id} with SR {sr_index}: {json_err}")
            print(f"Response raw: {response[:500]}")
            result['status'] = 'failed'
            result['error'] = f"JSON parsing error: {str(json_err)}"
            
    except Exception as e:
        print(f"Worker {worker_id} failed processing function ID {function_id} with SR {sr_index}: {e}")
        result['status'] = 'failed'
        result['error'] = str(e)
    
    print(f"Worker {worker_id} completed function ID: {function_id} with SR index: {sr_index}")
    return result

def get_failed_entries(output_file):
    """Get entries that failed in previous run from results file."""
    if not os.path.exists(output_file):
        print(f"Results file {output_file} does not exist yet, no failed entries to retry.")
        return []
    
    try:
        results_df = pd.read_csv(output_file)
        failed_entries = results_df[results_df['status'] == 'failed']
        print(f"Found {len(failed_entries)} failed entries to retry.")
        return failed_entries.to_dict('records')
    except Exception as e:
        print(f"Error reading results file: {e}")
        return []

async def retry_failed_entries(failed_entries, function_df, sr_df, model_name, verify_prompt, max_workers=10):
    """Retry processing the failed entries."""
    if not failed_entries:
        return []
    
    print(f"Retrying {len(failed_entries)} failed entries")
    
    # Group failed entries into batches for workers
    max_concurrent_tasks = min(max_workers, len(failed_entries))
    chunk_size = len(failed_entries) // max_concurrent_tasks + (1 if len(failed_entries) % max_concurrent_tasks else 0)
    
    # Create tasks for each worker
    tasks = []
    for i in range(max_concurrent_tasks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(failed_entries))
        if start_idx >= len(failed_entries):
            break
        chunk = failed_entries[start_idx:end_idx]
        tasks.append(process_retry_chunk(i, chunk, function_df, sr_df, model_name, verify_prompt))
    
    # Wait for all tasks to complete and collect results
    retry_results = await asyncio.gather(*tasks)
    
    # Flatten results list
    return [item for sublist in retry_results for item in sublist]

async def process_retry_chunk(worker_id, entries, function_df, sr_df, model_name, verify_prompt):
    """Process a chunk of failed entries for retry."""
    client = AsyncArk(api_key="xxx")
    endpoint = get_endpoint(model_name)
    results = []
    
    print(f"Retry worker {worker_id} processing {len(entries)} failed entries")
    
    for entry in entries:
        function_id = entry['function_id']
        sr_index = entry['sr_index']
        
        # Find the function in function_df
        function_row = function_df[function_df['function_id'] == function_id]
        if function_row.empty:
            print(f"Retry worker {worker_id}: Function ID {function_id} not found in function dataframe")
            entry['status'] = 'failed'
            entry['error'] = "Function not found in source dataframe"
            results.append(entry)
            continue
            
        function_name = function_row['function_name'].iloc[0]
        source_code = function_row['source_code'].iloc[0]
        
        # Find the SR in sr_df
        try:
            idx = int(sr_index)
            sr_row = sr_df.iloc[[idx]]
            
            # Get SR Text as the primary SR content
            if 'SR Text' in sr_row.columns:
                sr_content = sr_row['SR Text'].iloc[0]
            else:
                sr_content = sr_row['Current Paragraph'].iloc[0]
            
            # Concatenate context for LLM prompt
            sr_context = ""
            if 'Previous Paragraph' in sr_row.columns and not pd.isna(sr_row['Previous Paragraph'].iloc[0]):
                sr_context += sr_row['Previous Paragraph'].iloc[0] + "\n\n"
            if 'Current Paragraph' in sr_row.columns and not pd.isna(sr_row['Current Paragraph'].iloc[0]):
                sr_context += sr_row['Current Paragraph'].iloc[0] + "\n\n"
            if 'Next Paragraph' in sr_row.columns and not pd.isna(sr_row['Next Paragraph'].iloc[0]):
                sr_context += sr_row['Next Paragraph'].iloc[0]
            
            # Retry the verification
            result = await verify_function_sr_pair(
                function_id,
                function_name,
                source_code,
                sr_content,
                sr_context,
                sr_index,
                client,
                endpoint,
                verify_prompt,
                f"Retry-{worker_id}"
            )
            results.append(result)
            
        except Exception as e:
            print(f"Retry worker {worker_id} error processing SR ID {sr_index}: {e}")
            entry['status'] = 'failed'
            entry['error'] = f"Retry error: {str(e)}"
            results.append(entry)
    
    print(f"Retry worker {worker_id} completed {len(results)} entries")
    return results

async def main():
    """Main function to run the verification processing."""
    args = parse_args()
    
    # TLS specific file paths
    function_file = '/data/a/ykw/RFC/final/data/boringssl/dual_filter_with_sr_ids.csv'
    sr_file = '/data/a/ykw/RFC/final/sr/tls/8446sr_with_keywords.csv'
    output_file = '/data/a/ykw/RFC/final/data/boringssl/pre_verify_results.csv'
    
    # Load prompt
    verify_prompt = load_prompt()
    
    # Read the function CSV file
    function_df = pd.read_csv(function_file)
    # function_df = function_df.head(10)
    print(f"Processing {len(function_df)} functions from {function_file}")
    
    # Read the SR CSV file - do not set index_col
    sr_df = pd.read_csv(sr_file)
    print(f"Loaded {len(sr_df)} SR entries from {sr_file}")
    print(f"SR dataframe columns: {sr_df.columns.tolist()}")
    
    # Check the first few rows to understand structure
    print("First SR row sample:")
    print(sr_df.iloc[0])
    
    # Check if we need to retry failed entries
    if args.retry:
        # Get failed entries from previous run
        failed_entries = get_failed_entries(output_file)
        
        if failed_entries:
            # Determine which model to use for retry
            retry_model = args.retry_model if args.retry_model else args.m
            # Determine how many workers to use for retry
            retry_workers = args.retry_workers if args.retry_workers else args.w
            
            print(f"Using model '{retry_model}' with {retry_workers} workers for retrying failed entries")
            
            # Retry processing failed entries
            retry_results = await retry_failed_entries(
                failed_entries, 
                function_df, 
                sr_df, 
                retry_model, 
                verify_prompt,
                retry_workers
            )
            
            if os.path.exists(output_file):
                # Read existing results
                existing_results = pd.read_csv(output_file)
                
                # Convert retry results to DataFrame
                retry_df = pd.DataFrame(retry_results)
                
                # Create a unique key for merging based on function_id and sr_index
                existing_results['unique_key'] = existing_results['function_id'].astype(str) + '_' + existing_results['sr_index'].astype(str)
                retry_df['unique_key'] = retry_df['function_id'].astype(str) + '_' + retry_df['sr_index'].astype(str)
                
                # Remove the failed entries that were retried from existing results
                existing_unique_keys = set(existing_results['unique_key'])
                retry_unique_keys = set(retry_df['unique_key'])
                keys_to_update = existing_unique_keys.intersection(retry_unique_keys)
                
                # Filter out rows to be replaced
                remaining_results = existing_results[~existing_results['unique_key'].isin(keys_to_update)]
                
                # Combine remaining results with retry results
                combined_df = pd.concat([remaining_results, retry_df], ignore_index=True)
                
                # Remove the temporary unique_key column
                combined_df = combined_df.drop(columns=['unique_key'])
                
                # Save the updated results
                combined_df.sort_values(by='function_id').to_csv(output_file, index=False)
                print(f"Updated results file with {len(retry_results)} retried entries")
                
                # If only retrying, exit here
                if args.retry_only:
                    print("Retry only mode - exiting after retrying failed entries")
                    return
            else:
                # If results file doesn't exist but we have retry results, save them
                if retry_results:
                    pd.DataFrame(retry_results).to_csv(output_file, index=False)
                    print(f"Created new results file with {len(retry_results)} retried entries")
                    
                # If only retrying, exit here
                if args.retry_only:
                    print("Retry only mode - exiting after retrying failed entries")
                    return
    
    # If retry_only is set and we reached here, there were no failed entries to retry
    if args.retry_only:
        print("Retry only mode - no failed entries found, exiting")
        return
                
    # Prepare all function-SR pairs
    all_pairs = []
    for _, row in function_df.iterrows():
        function_id = row['function_id']
        function_name = row['function_name']
        source_code = row['source_code']
        sr_ids = row['matching_sr_ids'] if 'matching_sr_ids' in row else []
        
        # Ensure sr_ids is always a list of strings
        if sr_ids is None:
            sr_ids = []
        elif isinstance(sr_ids, int) or isinstance(sr_ids, float):
            sr_ids = [str(int(sr_ids))]
        elif isinstance(sr_ids, str):
            # Try parsing as JSON first
            try:
                sr_ids = json.loads(sr_ids)
                # If the loaded result is not a list, convert it to a list
                if not isinstance(sr_ids, list):
                    sr_ids = [str(sr_ids)]
            except json.JSONDecodeError:
                # If not JSON, split by comma
                sr_ids = [id.strip() for id in sr_ids.split(',') if id.strip()]
        
        # Ensure sr_ids is a list at this point
        if not isinstance(sr_ids, list):
            sr_ids = [str(sr_ids)]
            
        # Create a separate task for each function-SR pair
        print(f"Processing function ID: {function_id} with SR IDs: {sr_ids}")

        for sr_id in sr_ids:
            # Make sure sr_id is a string and not empty
            sr_id = str(sr_id).strip()
            if not sr_id:
                continue
                
            # Ensure sr_id is a valid index
            try:
                idx = int(sr_id)
                if idx < len(sr_df):
                    all_pairs.append({
                        'function_id': function_id,
                        'function_name': function_name,
                        'source_code': source_code,
                        'sr_id': sr_id
                    })
                else:
                    print(f"Warning: SR ID {sr_id} is out of range in SR dataframe")
            except ValueError:
                print(f"Warning: SR ID {sr_id} cannot be converted to integer")
    
    print(f"Total function-SR pairs to process: {len(all_pairs)}")
    
    # Check if we have already processed some pairs
    processed_pairs = set()
    if os.path.exists(output_file) and not args.force:
        try:
            existing_results = pd.read_csv(output_file)
            # Create unique identifiers for already processed pairs
            for _, row in existing_results.iterrows():
                if row['status'] == 'success':  # Only skip successful entries
                    key = f"{row['function_id']}_{row['sr_index']}"
                    processed_pairs.add(key)
            
            print(f"Found {len(processed_pairs)} already successfully processed pairs")
        except Exception as e:
            print(f"Error reading existing results file: {e}")
    
    # Filter out already processed pairs
    if processed_pairs and not args.force:
        filtered_pairs = []
        for pair in all_pairs:
            key = f"{pair['function_id']}_{pair['sr_id']}"
            if key not in processed_pairs:
                filtered_pairs.append(pair)
        
        skipped_count = len(all_pairs) - len(filtered_pairs)
        print(f"Skipping {skipped_count} already processed pairs")
        all_pairs = filtered_pairs
    
    if not all_pairs:
        print("No new pairs to process")
        return
    
    # Configure batch processing
    max_concurrent_tasks = args.w
    chunk_size = len(all_pairs) // max_concurrent_tasks + (1 if len(all_pairs) % max_concurrent_tasks else 0)
    
    start = datetime.now()
    
    # Split pairs into chunks and create tasks
    tasks = []
    for i in range(max_concurrent_tasks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(all_pairs))
        if start_idx >= len(all_pairs):
            break
        chunk = all_pairs[start_idx:end_idx]
        tasks.append(process_pairs(i, chunk, sr_df, args.m, verify_prompt))

    # Wait for all tasks to complete and collect results
    all_results = await asyncio.gather(*tasks)
    
    # Flatten results list
    final_results = [item for sublist in all_results for item in sublist]
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(final_results)
    
    # If the output file exists, merge with existing results
    if os.path.exists(output_file) and not args.force:
        try:
            existing_df = pd.read_csv(output_file)
            # Concatenate with new results
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            # Remove duplicates based on function_id and sr_index, keeping the latest version
            combined_df = combined_df.drop_duplicates(subset=['function_id', 'sr_index'], keep='last')
            # Sort and save
            combined_df.sort_values(by='function_id').to_csv(output_file, index=False)
            print(f"Updated existing results with {len(final_results)} new entries")
        except Exception as e:
            print(f"Error when trying to merge with existing results: {e}")
            # Fallback: save just the new results
            results_df.sort_values(by='function_id').to_csv(output_file, index=False)
    else:
        # Save the results to a new CSV
        results_df.sort_values(by='function_id').to_csv(output_file, index=False)
        print(f"Created new results file with {len(final_results)} entries")
    
    end = datetime.now()
    print(f"Total time: {end - start}")
    print(f"Total function-SR pairs processed: {len(final_results)}")
    print(f"Results saved to {output_file}")

async def process_pairs(worker_id, pairs, sr_df, model_name, verify_prompt):
    """Process a chunk of function-SR pairs."""
    client = AsyncArk(api_key="d4a025cf-edee-4a72-ae86-1139e9ef055e")
    endpoint = get_endpoint(model_name)
    results = []
    
    print(f"Worker {worker_id} processing {len(pairs)} function-SR pairs")
    
    for pair in pairs:
        function_id = pair['function_id']
        function_name = pair['function_name']
        source_code = pair['source_code']
        sr_id = pair['sr_id']
        
        # Find the SR in the sr_df by matching the row index/id
        try:
            idx = int(sr_id)
            sr_row = sr_df.iloc[[idx]]
            
            # Get SR Text as the primary SR content
            if 'SR Text' in sr_row.columns:
                sr_content = sr_row['SR Text'].iloc[0]
            else:
                sr_content = sr_row['Current Paragraph'].iloc[0]
            
            # Concatenate context for LLM prompt
            sr_context = ""
            if 'Previous Paragraph' in sr_row.columns and not pd.isna(sr_row['Previous Paragraph'].iloc[0]):
                sr_context += sr_row['Previous Paragraph'].iloc[0] + "\n\n"
            if 'Current Paragraph' in sr_row.columns and not pd.isna(sr_row['Current Paragraph'].iloc[0]):
                sr_context += sr_row['Current Paragraph'].iloc[0] + "\n\n"
            if 'Next Paragraph' in sr_row.columns and not pd.isna(sr_row['Next Paragraph'].iloc[0]):
                sr_context += sr_row['Next Paragraph'].iloc[0]
            
            result = await verify_function_sr_pair(
                function_id,
                function_name,
                source_code,
                sr_content,
                sr_context,
                sr_id,
                client,
                endpoint,
                verify_prompt,
                worker_id
            )
            results.append(result)
            
        except Exception as e:
            print(f"Worker {worker_id} error processing SR ID {sr_id}: {e}")
    
    print(f"Worker {worker_id} completed {len(results)} function-SR pairs")
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Verify TLS functions against SR requirements')
    parser.add_argument('-m', type=str, choices=['pro', 'lite', 'dsv3', 'dsr1'], 
                       default='dsr1', help='Model name to use (pro, lite, dsv3, or dsr1)')
    parser.add_argument('-w', type=int, default=20,
                       help='Number of concurrent workers (default: 20)')
    parser.add_argument('--retry', action='store_true',
                       help='Retry failed requests from previous run before processing new ones')
    parser.add_argument('--retry-only', action='store_true',
                       help='Only retry failed requests from previous run, don\'t process new ones')
    parser.add_argument('--retry-model', type=str, choices=['pro', 'lite', 'dsv3', 'dsr1'],
                       help='Model name to use specifically for retries (defaults to -m if not specified)')
    parser.add_argument('--retry-workers', type=int,
                       help='Number of concurrent workers for retries (defaults to -w if not specified)')
    parser.add_argument('--force', action='store_true',
                       help='Force processing all pairs, ignoring previous results')
    return parser.parse_args()

if __name__ == "__main__":
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(main())
    else:
        uvloop.install()
        asyncio.run(main())
