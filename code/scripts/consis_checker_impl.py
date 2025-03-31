import os
import sys
import json
import asyncio
import pandas as pd
import uvloop
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import argparse
from math import ceil
import re
from volcenginesdkarkruntime import AsyncArk

# 直接复制consis_checker.py中的prompt
user_prompt_template = """You are a software verification expert specializing in protocol implementations. Your task is to analyze whether the given code implementation is meant to, and does in fact, implement the specification requirement from a protocol RFC. If it does implement the requirement, evaluate whether it does so correctly and completely.

### Specification Requirement:
<specification>
{SR_text}
</specification>

### Implementation:
<implementation>
{function_code}
</implementation>

### Context:
<call_graph>
Caller functions: {caller_functions}
Callee functions: {callee_functions}
</call_graph>

<variables>
Global variables used: {global_variables}
Macro definitions: {macros}
</variables>

### Verification Task:
1. **Determine if the implementation is directly addressing the specification requirement** stated above. If the code is not intended to implement that specific requirement, explain why. This is a critical first step - you must determine if this code is actually supposed to implement the given spec.

2. **If the code is indeed implementing the requirement**, carefully analyze whether it correctly handles the trigger condition specified in the requirement:
   - Does it detect or handle the relevant condition(s) as intended by the spec?

3. **Verify whether the implementation performs the required action** when the condition is met:
   - Are there any deviations from the specification's prescribed behavior?

4. **Check for edge cases or exceptions** that might cause the implementation to violate the specification.

5. **Examine the relevant state transitions** to ensure they adhere to the protocol's expected behavior.

### Output Format:
Provide your verification result in JSON format with the following structure:
```json
{{
  "implements_spec": "YES|NO|UNKNOWN",
  "conformance": "FULL|PARTIAL|NONE",
  "reasoning": "Detailed step-by-step analysis explaining your reasoning",
  "issues": [
    {{
      "description": "Description of the inconsistency or bug",
      "code_segment": "The specific code segment that contains the issue",
      "line_numbers": "Approximate line numbers in the provided implementation",
      "expected_behavior": "What the code should do according to the specification",
      "actual_behavior": "What the code actually does",
      "severity": "HIGH|MEDIUM|LOW"
    }}
  ],
  "conclusion": "Summary of the verification result"
}}

IMPORTANT: You MUST set "implements_spec" to one of:
- "YES" if the code is clearly implementing the specification requirement
- "NO" if the code is not implementing this specific requirement
- "UNKNOWN" if you cannot determine whether the code implements the requirement

For "conformance", set it to:
- "FULL" if the code fully implements the spec
- "PARTIAL" if it partly implements the spec but has issues
- "NONE" if it completely fails to implement the spec correctly
- If implements_spec is "NO", set conformance to "NONE"
```"""

class ConsistencyResult(BaseModel):
    """Pydantic model for consistency check results"""
    implements_spec: str = Field(..., description="YES|NO|UNKNOWN")
    conformance: str = Field(..., description="FULL|PARTIAL|NONE")
    reasoning: str = Field(..., description="Detailed step-by-step analysis explaining your reasoning")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="List of identified issues")
    conclusion: str = Field(..., description="Summary of the verification result") 

async def check_consistency(function_id: int, function_name: str, sr_content: str, sr_index: int,
                           source_code: str, dependencies: str, file_path: str,
                           sr_context: Dict[str, str], client: AsyncOpenAI) -> Dict[str, Any]:
    """Check consistency between a function implementation and specification requirement"""
    print(f"Checking consistency for function ID: {function_id}")
    
    # Extract useful dependency information
    useful_dependencies = extract_useful_dependencies(dependencies)
    
    # Extract caller and callee functions from dependencies
    caller_functions = []
    callee_functions = []
    global_vars = []
    macros = []
    
    try:
        deps_json = json.loads(dependencies)
        # Extract caller functions
        if "callers" in deps_json:
            caller_functions = list(deps_json["callers"].keys())
        
        # Extract callee functions
        if "called_functions" in deps_json:
            callee_functions = list(deps_json["called_functions"].keys())
        
        # Extract global variables
        if "used_globals" in deps_json:
            global_vars = list(deps_json["used_globals"].keys())
        
        # Extract macros
        if "used_macros" in deps_json:
            macros = list(deps_json["used_macros"].keys())
            
    except Exception as e:
        print(f"Error parsing dependencies for function ID {function_id}: {e}")
    
    # Format the prompt with the actual data
    formatted_prompt = user_prompt_template.format(
        SR_text=sr_content,
        function_code=source_code,
        caller_functions=", ".join(caller_functions),
        callee_functions=", ".join(callee_functions),
        global_variables=", ".join(global_vars),
        macros=", ".join(macros)
    )
    
    try:
        # Call the OpenAI API to check consistency
        completion = await client.chat.completions.create(
            model="o3-mini",  # Use an appropriate model
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        response_text = completion.choices[0].message.content
        response_json = json.loads(response_text)
        
        print(f"Received consistency check response for function ID: {function_id}")
        
        # Handle the case where the implementation doesn't address the specification
        implements_spec = response_json.get("implements_spec", "UNKNOWN")
        if implements_spec == "NO":
            print(f"Function {function_id} does not implement the specification. Skipping detailed consistency check.")
            # Set conformance to NONE if the code doesn't implement the spec
            if "conformance" not in response_json:
                response_json["conformance"] = "NONE"
        
        # Combine all the data we want to save - ensuring no dependency information is included
        result = {
            'function_id': function_id,
            'function_name': function_name,
            'file_path': file_path,
            'sr_content': sr_content,
            'sr_index': sr_index,
            'consistency_result': response_json,
            'status': 'success',
            'function_code': source_code,
            'sr_context': sr_context,
        }
        
        return result
        
    except Exception as e:
        print(f"Failed checking consistency for function ID {function_id}: {e}")
        
        return {
            'function_id': function_id,
            'function_name': function_name,
            'file_path': file_path,
            'sr_content': sr_content,
            'sr_index': sr_index,
            'consistency_result': None,
            'status': 'failed',
            'error': str(e),
            'function_code': source_code,
            'sr_context': sr_context,
        }

async def check_consistency_volcengine(function_id: int, function_name: str, sr_content: str, sr_index: int,
                           source_code: str, dependencies: str, file_path: str,
                           sr_context: Dict[str, str], client, endpoint: str) -> Dict[str, Any]:
    """Check consistency between a function implementation and specification requirement using volcengine models"""
    print(f"Checking consistency using volcengine for function ID: {function_id}")
    
    # Extract useful dependency information
    useful_dependencies = extract_useful_dependencies(dependencies)
    
    # Extract caller and callee functions from dependencies
    caller_functions = []
    callee_functions = []
    global_vars = []
    macros = []
    
    try:
        deps_json = json.loads(dependencies)
        # Extract caller functions
        if "callers" in deps_json:
            caller_functions = list(deps_json["callers"].keys())
        
        # Extract callee functions
        if "called_functions" in deps_json:
            callee_functions = list(deps_json["called_functions"].keys())
        
        # Extract global variables
        if "used_globals" in deps_json:
            global_vars = list(deps_json["used_globals"].keys())
        
        # Extract macros
        if "used_macros" in deps_json:
            macros = list(deps_json["used_macros"].keys())
            
    except Exception as e:
        print(f"Error parsing dependencies for function ID {function_id}: {e}")
    
    # Format the prompt with the actual data
    formatted_prompt = user_prompt_template.format(
        SR_text=sr_content,
        function_code=source_code,
        caller_functions=", ".join(caller_functions),
        callee_functions=", ".join(callee_functions),
        global_variables=", ".join(global_vars),
        macros=", ".join(macros)
    )
    
    try:
        # Call the Volcengine API to check consistency
        completion = await client.chat.completions.create(
            model=endpoint,
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
            max_tokens=1000,
            temperature=0
        )
        
        res = completion.choices[0].message.content
        print(f"Received volcengine response for function ID {function_id}: {res[:100]}...")
        
        try:
            # Extract JSON content if it's wrapped in markdown code blocks
            json_content = extract_json_from_response(res)
            
            response_json = json.loads(json_content)
            
            print(f"Parsed JSON response for function ID: {function_id}")
            
            # Handle the case where the implementation doesn't address the specification
            implements_spec = response_json.get("implements_spec", "UNKNOWN")
            if implements_spec == "NO":
                print(f"Function {function_id} does not implement the specification. Skipping detailed consistency check.")
                # Set conformance to NONE if the code doesn't implement the spec
                if "conformance" not in response_json:
                    response_json["conformance"] = "NONE"
            
            # Combine all the data we want to save - ensuring no dependency information is included
            result = {
                'function_id': function_id,
                'function_name': function_name,
                'file_path': file_path,
                'sr_content': sr_content,
                'sr_index': sr_index,
                'consistency_result': response_json,
                'status': 'success',
                'function_code': source_code,
                'sr_context': sr_context,
            }
            
            return result
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing error for function ID {function_id}: {json_err}")
            print(f"Raw response: {res[:500]}")
            return {
                'function_id': function_id,
                'function_name': function_name,
                'file_path': file_path,
                'sr_content': sr_content,
                'sr_index': sr_index,
                'consistency_result': None,
                'status': 'failed',
                'error': f"JSON parsing error: {res[:500]}",
                'function_code': source_code,
                'sr_context': sr_context,
            }
        
    except Exception as e:
        print(f"Failed checking consistency for function ID {function_id}: {e}")
        
        return {
            'function_id': function_id,
            'function_name': function_name,
            'file_path': file_path,
            'sr_content': sr_content,
            'sr_index': sr_index,
            'consistency_result': None,
            'status': 'failed',
            'error': str(e),
            'function_code': source_code,
            'sr_context': sr_context,
        }

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
        return "ep-20250210193928-fl84t"  # ds r1 dstill
    else:
        raise ValueError(f"Invalid model name: {model_name}")

async def worker(worker_id: int, pairs_to_check: pd.DataFrame, 
                func_data_df: pd.DataFrame, sr_df: pd.DataFrame,
                model_type: str = "openai") -> List[Dict[str, Any]]:
    """Worker function to process assigned pairs"""
    
    # Initialize appropriate client based on model type
    if model_type in ["pro", "lite", "dsv3", "dsr1"]:
        # Use volcengine for these models
        client = AsyncArk(api_key="xxx") # Or get from environment variable
        endpoint = get_endpoint(model_type)
    else:
        # Use OpenAI for "openai" model
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = AsyncOpenAI(api_key=api_key)
    
    results = []
    
    total_pairs = len(pairs_to_check)
    print(f"Worker {worker_id} starting processing {total_pairs} pairs with model {model_type}")
    
    for idx, row in enumerate(pairs_to_check.iterrows(), 1):
        row = row[1]  # Get the actual row data
        function_id = row['function_id']
        function_name = row.get('function_name', '')
        sr_content = row['sr_content']
        sr_index = row['sr_index']
        
        # Find the corresponding function data in func_data_df
        function_row = func_data_df[func_data_df['function_id'] == function_id]
        if function_row.empty:
            print(f"Warning: Function ID {function_id} not found in function data")
            continue
        
        # Extract source code, dependencies and file path
        source_code = function_row.iloc[0]['source_code'] if 'source_code' in function_row else ''
        dependencies = function_row.iloc[0]['dependencies'] if 'dependencies' in function_row else '{}'
        file_path = function_row.iloc[0]['file_path'] if 'file_path' in function_row else ''
        
        # Get SR context from sr_df using sr_index
        sr_context = {}
        try:
            sr_row = sr_df.iloc[sr_index]
            for field in ['Previous Paragraph', 'Current Paragraph', 'Next Paragraph']:
                if field in sr_row and pd.notna(sr_row.get(field)) and sr_row.get(field):
                    sr_context[field] = sr_row[field]
        except Exception as e:
            print(f"Error getting SR context for SR index {sr_index}: {e}")
        
        # Use appropriate consistency check function based on model type
        if model_type in ["pro", "lite", "dsv3", "dsr1"]:
            result = await check_consistency_volcengine(
                function_id=function_id,
                function_name=function_name,
                sr_content=sr_content,
                sr_index=sr_index,
                source_code=source_code,
                dependencies=dependencies,
                file_path=file_path,
                sr_context=sr_context,
                client=client,
                endpoint=endpoint
            )
        else:
            result = await check_consistency(
                function_id=function_id,
                function_name=function_name,
                sr_content=sr_content,
                sr_index=sr_index,
                source_code=source_code,
                dependencies=dependencies,
                file_path=file_path,
                sr_context=sr_context,
                client=client
            )
        
        results.append(result)
        
        # Display progress for this worker
        print(f"Worker {worker_id} completed {idx}/{total_pairs} ({idx/total_pairs*100:.1f}%)")

    print(f"Worker {worker_id} completed all {total_pairs} pairs")
    return results

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RFC Consistency Checker')
    parser.add_argument('-w', '--workers', type=int, default=None,
                        help='Number of workers to use (default: auto-detect based on CPU cores)')
    parser.add_argument('-m', '--model', type=str, choices=['pro', 'lite', 'dsv3', 'dsr1', 'openai'], 
                        default='openai', help='Model to use (pro, lite, dsv3, dsr1, or openai)')
    parser.add_argument('-p', '--protocol', type=str, choices=['http', 'tls', 'HTTP', 'TLS'],
                        default='HTTP', help='Protocol to use (HTTP or TLS)')
    
    args = parser.parse_args()
    
    # Normalize protocol to uppercase
    protocol = args.protocol.upper()
    
    # Set file paths based on protocol
    if protocol == 'HTTP':
        # HTTP protocol paths
        fully_matches_csv = "/data/a/ykw/RFC/final/data/nginx/fully_matches.csv"
        func_data_csv = "/data/a/ykw/RFC/final/data/nginx/dual_filter_with_sr_ids.csv"
        sr_csv = "/data/a/ykw/RFC/final/sr/msm_filtered.csv"
        output_dir = "/data/a/ykw/RFC/final/data/nginx"
    else:  # TLS
        # TLS protocol paths
        fully_matches_csv = "/data/a/ykw/RFC/final/data/openssl/fully_matches.csv"
        func_data_csv = "/data/a/ykw/RFC/final/data/openssl/dual_filter_with_sr_ids.csv"
        sr_csv = "/data/a/ykw/RFC/final/sr/tls/8446sr_with_keywords.csv"
        output_dir = "/data/a/ykw/RFC/final/data/openssl"
    
    # Set output file path with protocol and model info
    output_jsonl = f"{output_dir}/consistency_results_{protocol.lower()}_{args.model}.jsonl"
    
    # Load all required data
    print("Loading data files...")
    try:
        fully_matches_df = pd.read_csv(fully_matches_csv)
        func_data_df = pd.read_csv(func_data_csv)
        sr_df = pd.read_csv(sr_csv)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return
    
    print(f"Found {len(fully_matches_df)} matched function-spec pairs to check")
    
    # Configure parallel processing
    if args.workers:
        num_workers = min(args.workers, len(fully_matches_df))
        print(f"Using {num_workers} workers as specified by user")
    else:
        num_workers = min(os.cpu_count() or 4, len(fully_matches_df))
        print(f"Auto-detected {num_workers} workers based on system configuration")
    
    # Calculate how many pairs each worker processes
    # fully_matches_df = fully_matches_df.head(10)
    total_pairs = len(fully_matches_df)
    pairs_per_worker = ceil(total_pairs / num_workers)
    
    start_time = datetime.now()
    print(f"Starting consistency checking with model: {args.model}")
    
    # Distribute data evenly among workers
    tasks = []
    for i in range(num_workers):
        start_idx = i * pairs_per_worker
        end_idx = min((i + 1) * pairs_per_worker, total_pairs)
        
        if start_idx >= total_pairs:
            break
            
        worker_data = fully_matches_df.iloc[start_idx:end_idx]
        tasks.append(worker(i, worker_data, func_data_df, sr_df, args.model))
    
    # Wait for all workers to complete and collect results
    all_results = await asyncio.gather(*tasks)
    
    # Merge all results
    flat_results = [result for worker_results in all_results for result in worker_results]
    
    # Generate a summary of the results
    implements_spec_counts = {"YES": 0, "NO": 0, "UNKNOWN": 0}
    conformance_counts = {"FULL": 0, "PARTIAL": 0, "NONE": 0, "UNKNOWN": 0}
    
    for result in flat_results:
        if result["status"] == "success" and result["consistency_result"]:
            implements_spec = result["consistency_result"].get("implements_spec", "UNKNOWN")
            implements_spec_counts[implements_spec] = implements_spec_counts.get(implements_spec, 0) + 1
            
            conformance = result["consistency_result"].get("conformance", "UNKNOWN")
            conformance_counts[conformance] = conformance_counts.get(conformance, 0) + 1
    
    # Write results to JSONL file - 使用漂亮的格式以提高可读性
    with open(output_jsonl, 'w') as f:
        for result in flat_results:
            # 使用indent参数使JSON更易读
            f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            f.write('\n')  # 添加空行分隔不同的结果
    
    # 同时生成一个常规的JSON文件，更容易用文本编辑器查看
    pretty_json_path = output_jsonl.replace('.jsonl', '_pretty.json')
    with open(pretty_json_path, 'w', encoding='utf-8') as f:
        json.dump(flat_results, f, ensure_ascii=False, indent=2)
    
    # Generate a summary file
    summary_path = output_jsonl.replace('.jsonl', '_summary.json')
    summary = {
        "total_pairs": len(flat_results),
        "success_count": sum(1 for r in flat_results if r["status"] == "success"),
        "failed_count": sum(1 for r in flat_results if r["status"] == "failed"),
        "implements_spec_distribution": implements_spec_counts,
        "conformance_distribution": conformance_counts
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nTotal consistency checking time: {elapsed_time}")
    print(f"Results saved to {output_jsonl}")
    print(f"Pretty formatted results saved to {pretty_json_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Average time per pair: {elapsed_time / len(flat_results) if flat_results else 0}")
    
    # Print summary to console
    print("\nSummary:")
    print(f"Total pairs: {len(flat_results)}")
    print(f"Successful checks: {summary['success_count']}")
    print(f"Failed checks: {summary['failed_count']}")
    print("\nImplements Spec Distribution:")
    for key, value in implements_spec_counts.items():
        print(f"  {key}: {value}")
    print("\nConformance Distribution:")
    for key, value in conformance_counts.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            runner.run(main())
    else:
        uvloop.install()
        asyncio.run(main()) 