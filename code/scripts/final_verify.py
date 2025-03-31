import sys
import os
import json
import pandas as pd
import uvloop
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from math import ceil

class ConstraintDetail(BaseModel):
    """Model for the details of a constraint match"""
    code: str = Field(..., description="Code segment that matches the constraint")
    explanation: str = Field(..., description="Explanation of the match")

class VerificationResult(BaseModel):
    """Pydantic model for verification results"""
    outcome: str = Field(..., description="Verification result: Fully Matches, Partially Matches, Does Not Match")
    met_constrains: List[Dict[str, ConstraintDetail]] = Field(default_factory=list, description="List of constraint matches in format [{C1: {code: '...', explanation: '...'}}, {A1: {code: '...', explanation: '...'}}]")
    OverallExplanation: str = Field(..., description="Overall explanation")


def extract_useful_dependencies(dependencies_str: str) -> str:
    """
    Extract useful function definitions and struct definitions from dependency information,
    ignoring other metadata.
    """
    try:
        # Parse dependency data
        dependencies = json.loads(dependencies_str)
        
        # Extract useful function definitions
        useful_deps = []
        
        # Process called_functions
        if "called_functions" in dependencies:
            for func_name, func_info in dependencies["called_functions"].items():
                if func_info.get("definition") and func_info["definition"] != "null":
                    useful_deps.append(f"// Function: {func_name}\n{func_info['definition']}")
        
        # Process struct definitions
        if "used_structs" in dependencies:
            for struct_name, struct_info in dependencies["used_structs"].items():
                if struct_info.get("definition") and struct_info["definition"] != "null":
                    useful_deps.append(f"// Struct: {struct_name}\n{struct_info['definition']}")
        
        # Process global variables
        if "used_globals" in dependencies:
            for global_name, global_info in dependencies["used_globals"].items():
                if global_info.get("full_definition") and global_info["full_definition"] != "null":
                    useful_deps.append(f"// Global: {global_name}\n{global_info['full_definition']}")

        # Process macro definitions
        if "used_macros" in dependencies:
            for macro_name, macro_info in dependencies["used_macros"].items():
                if macro_info.get("definition") and macro_info["definition"] != "null":
                    useful_deps.append(f"// Macro: {macro_name}\n{macro_info['definition']}")
                elif macro_info.get("value") and macro_info["value"] != "null":
                    # If there is no complete definition but there is a value, create a simple definition
                    useful_deps.append(f"// Macro: {macro_name}\n#define {macro_name} {macro_info['value']}")

        # Process caller functions
        if "callers" in dependencies:
            for caller_name, caller_info in dependencies["callers"].items():
                # Clean caller name, remove location information
                clean_name = caller_name.split('@')[0] if '@' in caller_name else caller_name
                if caller_info.get("source_code") and caller_info["source_code"] != "null":
                    useful_deps.append(f"// Caller: {clean_name}\n{caller_info['source_code']}")
        
        return "\n\n".join(useful_deps)
    except Exception as e:
        print(f"Error extracting dependencies: {e}")
        return "// Failed to parse dependencies"

async def verify_pair(function_id: int, function_name: str, sr_content: str, sr_index: int,
                protocol: str, source_code: str, dependencies: str, sr_context: str,
                conditions: List[str], actions: List[str],
                client: AsyncOpenAI, verify_prompts: Dict[str, str]) -> Dict[str, Any]:
    """Verify a single code-spec pair against RFC requirements"""
    print(f"Verifying function ID: {function_id} with SR index: {sr_index}")
    
    # Extract useful dependency information
    useful_dependencies = extract_useful_dependencies(dependencies)
    
    # Format conditions and actions with C1, C2, A1, A2 style labels
    formatted_conditions = [f"C{i+1}: {cond}" for i, cond in enumerate(conditions)]
    formatted_actions = [f"A{i+1}: {act}" for i, act in enumerate(actions)]
    
    # Prepare the data for prompt
    spec_constraints = f"Conditions: {', '.join(formatted_conditions)}\nActions: {', '.join(formatted_actions)}"
    
    # Select the appropriate prompt based on protocol
    if protocol.lower() == 'http':
        verify_prompt = verify_prompts.get('http', '')
    elif protocol.lower() == 'tls':
        verify_prompt = verify_prompts.get('tls', '')
    else:
        # Default to HTTP if unknown
        verify_prompt = verify_prompts.get('http', '')
    
    # Replace placeholders directly instead of using format()
    prompt_with_data = verify_prompt
    prompt_with_data = prompt_with_data.replace("{spec}", sr_content)
    prompt_with_data = prompt_with_data.replace("{spec_constraints}", spec_constraints)
    prompt_with_data = prompt_with_data.replace("{sr_context}", sr_context)
    prompt_with_data = prompt_with_data.replace("{function_body}", source_code)
    prompt_with_data = prompt_with_data.replace("{dependencies}", useful_dependencies)
    
    # Add explicit instruction about constraint format
    prompt_with_data += """
    
IMPORTANT: Your response MUST be formatted exactly as follows:
{
    "outcome": "Fully Matches/Partially Matches/Does Not Match",
    "met_constrains": [
        {"C1": {"code": "actual code segment here", "explanation": "explanation here"}},
        {"A1": {"code": "actual code segment here", "explanation": "explanation here"}}
    ],
    "OverallExplanation": "overall explanation here"
}
"""
    
    try:
        # Use chat completions for better control over the output format
        completion = await client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are a professional code analysis and verification assistant. When referring to constraints, use only C1, C2, A1, A2 format as keys in JSON objects."},
                {"role": "user", "content": prompt_with_data},
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        response_text = completion.choices[0].message.content
        response_json = json.loads(response_text)
        
        print(f"Received verification response for function ID: {function_id}")
        
        # Extract data from response
        outcome = response_json.get("outcome", "Error")
        overall_explanation = response_json.get("OverallExplanation", "")
        
        # Process met_constrains to ensure the right format
        met_constrains = response_json.get("met_constrains", [])
        
        # Ensure each constraint is in the correct format: {"C1": {"code": "...", "explanation": "..."}}
        formatted_constraints = []
        for constraint_item in met_constrains:
            # Find the constraint label (C1, A1, etc.)
            constraint_keys = [k for k in constraint_item.keys() if k.startswith('C') or k.startswith('A')]
            
            if constraint_keys:
                for key in constraint_keys:
                    # Get the constraint content
                    content = constraint_item[key]
                    
                    # Handle both possible formats
                    if isinstance(content, dict) and "code" in content and "explanation" in content:
                        # Already in the correct format
                        formatted_constraint = {key: content}
                    else:
                        # Convert to the correct format
                        explanation = constraint_item.get("explanation", "")
                        formatted_constraint = {
                            key: {
                                "code": content if isinstance(content, str) else str(content),
                                "explanation": explanation
                            }
                        }
                    
                    formatted_constraints.append(formatted_constraint)
        
        return {
            'function_id': function_id,
            'function_name': function_name,
            'sr_content': sr_content,
            'sr_index': sr_index,
            'outcome': outcome,
            'met_constrains': formatted_constraints,
            'explanation': overall_explanation,
            'status': 'success',
            'api_response': None
        }
    except Exception as e:
        print(f"Failed verifying function ID {function_id}: {e}")
        
        return {
            'function_id': function_id,
            'function_name': function_name,
            'sr_content': sr_content,
            'sr_index': sr_index,
            'outcome': "Error",
            'met_constrains': [],
            'explanation': str(e),
            'status': 'failed',
            'api_response': str(e)
        }

def get_sr_context_and_constraints(sr_index: int, sr_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract SR context and constraints from SR dataframe"""
    result = {
        'sr_context': '',
        'conditions': [],
        'actions': []
    }
    
    # 直接使用索引号访问DataFrame
    try:
        # 确保sr_index是整数
        sr_index = int(sr_index)
        
        # 检查索引是否在有效范围内
        if 0 <= sr_index < len(sr_df):
            sr_row = sr_df.iloc[sr_index]
            print(f"Found SR at index position: {sr_index}")
        else:
            print(f"Warning: SR index {sr_index} out of bounds for dataframe with {len(sr_df)} rows")
            return result
    except Exception as e:
        print(f"Error accessing SR at index {sr_index}: {e}")
        return result
    
    # Combine context paragraphs
    context_parts = []
    for field in ['Previous Paragraph', 'Current Paragraph', 'Next Paragraph']:
        if field in sr_row and pd.notna(sr_row.get(field)) and sr_row.get(field):
            context_parts.append(sr_row[field])
    
    result['sr_context'] = " ".join(context_parts)
    
    # Extract conditions and actions
    if 'Conditions' in sr_row and pd.notna(sr_row.get('Conditions')):
        conditions_str = sr_row['Conditions']
        if isinstance(conditions_str, str) and ',' in conditions_str:
            result['conditions'] = [c.strip() for c in conditions_str.split(',')]
        else:
            result['conditions'] = [conditions_str]
    
    if 'Actions' in sr_row and pd.notna(sr_row.get('Actions')):
        actions_str = sr_row['Actions']
        if isinstance(actions_str, str) and ',' in actions_str:
            result['actions'] = [a.strip() for a in actions_str.split(',')]
        else:
            result['actions'] = [actions_str]
    
    return result

async def worker(worker_id: int, pairs_to_verify: pd.DataFrame, 
                dual_filter_df: pd.DataFrame, sr_df: pd.DataFrame, 
                verify_prompts: Dict[str, str], default_protocol: str) -> List[Dict[str, Any]]:
    """Worker function to process assigned pairs"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = AsyncOpenAI(api_key=api_key)
    results = []
    
    total_pairs = len(pairs_to_verify)
    print(f"Worker {worker_id} starting processing {total_pairs} pairs")
    
    for idx, row in enumerate(pairs_to_verify.iterrows(), 1):
        row = row[1]  # Get the actual row data
        function_id = row['function_id']
        function_name = row.get('function_name', '')
        sr_content = row['sr_content']
        sr_index = row['sr_index']
        
        # Find the corresponding function data in dual_filter_df
        function_row = dual_filter_df[dual_filter_df['function_id'] == function_id]
        if function_row.empty:
            print(f"Warning: Function ID {function_id} not found in function data")
            continue
        
        # Extract source code, dependencies and protocol
        source_code = function_row.iloc[0]['source_code'] if 'source_code' in function_row else ''
        dependencies = function_row.iloc[0]['dependencies'] if 'dependencies' in function_row else '{}'
        
        # Use default protocol specified by user instead of assuming HTTP
        protocol = function_row.iloc[0]['protocol'] if 'protocol' in function_row else default_protocol
        
        # Get SR context and constraints
        sr_info = get_sr_context_and_constraints(sr_index, sr_df)
        
        result = await verify_pair(
            function_id=function_id,
            function_name=function_name,
            sr_content=sr_content,
            sr_index=sr_index,
            protocol=protocol,
            source_code=source_code,
            dependencies=dependencies,
            sr_context=sr_info['sr_context'],
            conditions=sr_info['conditions'],
            actions=sr_info['actions'],
            client=client,
            verify_prompts=verify_prompts
        )
        
        results.append(result)
        
        # Display progress for this worker
        print(f"Worker {worker_id} completed {idx}/{total_pairs} ({idx/total_pairs*100:.1f}%)")

    print(f"Worker {worker_id} completed all {total_pairs} pairs")
    return results

async def main():
    # Parse command line arguments - only keep protocol and workers
    parser = argparse.ArgumentParser(description='RFC Verification Tool')
    parser.add_argument('-p', '--protocol', choices=['http', 'tls', 'HTTP', 'TLS'], default='HTTP',
                        help='Default protocol to use for verification (HTTP or TLS)')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of workers to use (default: auto-detect based on CPU cores)')
    
    args = parser.parse_args()
    
    # Normalize protocol to uppercase
    default_protocol = args.protocol.upper()
    
    # File paths - hardcoded
    pre_verify_results = "/data/a/ykw/RFC/final/data/boringssl/pre_verify_results.csv"
    dual_filter_csv = "/data/a/ykw/RFC/final/data/boringssl/dual_filter_with_sr_ids.csv"
    output_csv = f"/data/a/ykw/RFC/final/data/boringssl/final_verification_results.csv"
    
    # sr_csv = "/data/a/ykw/RFC/final/sr/msm_filtered.csv"
    if default_protocol == 'TLS':
        sr_csv = "/data/a/ykw/RFC/final/sr/tls/8446sr_with_keywords.csv"
    elif default_protocol == 'HTTP':
        sr_csv = "/data/a/ykw/RFC/final/sr/msm_filtered.csv"
    
    
    # Load verification prompts
    http_prompt_path = "/data/a/ykw/RFC/final/prompt/http_final_verify.txt"
    tls_prompt_path = "/data/a/ykw/RFC/final/prompt/tls_final_verify.txt"
    
    verify_prompts = {}
    
    try:
        with open(http_prompt_path, 'r') as f:
            verify_prompts['http'] = f.read()
        with open(tls_prompt_path, 'r') as f:
            verify_prompts['tls'] = f.read()
        print(f"Loaded verification prompts (HTTP: {len(verify_prompts['http'])} chars, TLS: {len(verify_prompts['tls'])} chars)")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return
    
    # Load all required data
    print("Loading data files...")
    try:
        pre_verify_df = pd.read_csv(pre_verify_results)
        dual_filter_df = pd.read_csv(dual_filter_csv)
        sr_df = pd.read_csv(sr_csv)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return
    
    # Filter for entries where is_match is True
    matched_pairs = pre_verify_df[pre_verify_df['is_match'] == True]
    # matched_pairs = matched_pairs.head(10)
    print(f"Found {len(matched_pairs)} matched function-spec pairs to verify")
    
    # Configure parallel processing
    if args.workers:
        num_workers = min(args.workers, len(matched_pairs))
        print(f"Using {num_workers} workers as specified by user")
    else:
        num_workers = min(os.cpu_count() or 4, len(matched_pairs))
        print(f"Auto-detected {num_workers} workers based on system configuration")
    
    # Calculate how many pairs each worker processes
    total_pairs = len(matched_pairs)
    pairs_per_worker = ceil(total_pairs / num_workers)
    # total_pairs = 10
    start_time = datetime.now()
    print(f"Starting verification with default protocol: {default_protocol}")
    
    # Distribute data evenly among workers
    tasks = []
    for i in range(num_workers):
        start_idx = i * pairs_per_worker
        end_idx = min((i + 1) * pairs_per_worker, total_pairs)
        
        if start_idx >= total_pairs:
            break
            
        worker_data = matched_pairs.iloc[start_idx:end_idx]
        tasks.append(worker(i, worker_data, dual_filter_df, sr_df, verify_prompts, default_protocol))
    
    # Wait for all workers to complete and collect results
    all_results = await asyncio.gather(*tasks)
    
    # Merge all results
    flat_results = [result for worker_results in all_results for result in worker_results]
    
    # Convert to DataFrame and save results
    results_df = pd.DataFrame(flat_results)
    results_df.to_csv(output_csv, index=False)
    
    # Print statistics
    if 'outcome' in results_df.columns:
        print("\nResult Statistics:")
        outcome_counts = results_df['outcome'].value_counts()
        for outcome, count in outcome_counts.items():
            print(f"Outcome '{outcome}': {count} pairs ({count/len(results_df)*100:.1f}%)")
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nTotal verification time: {elapsed_time}")
    print(f"Results saved to {output_csv}")
    print(f"Average time per pair: {elapsed_time / len(flat_results) if flat_results else 0}")

if __name__ == "__main__":
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(main())
    else:
        uvloop.install()
        asyncio.run(main())
