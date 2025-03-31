import asyncio
import json
import os
import csv
import time
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI

# Initialize async OpenAI client
client = AsyncOpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

class RoleJudgmentResponse(BaseModel):
    conditionlist: List[str]
    actionlist: List[str]

# Define async GPT API call function
async def call_gpt_api_async(sr, context):
    sys_prompt = f"""You are an expert of protocol. Extract and identify two structured semantic components from the provided Specification Requirement (SR) within the context of an HTTP protocol specification. These components are the Trigger Condition and the Required Action. Note that the Trigger Condition can be multiple and may also be empty, while the Required Action must always be present.

Ensure each component is clearly defined under separate headings. The Specification Requirement will be provided along with possible contextual paragraphs preceding and following it.

# Steps

1. **Review the Specification Requirement:** Carefully read the given SR and any provided context.
2. **Identify the Trigger Condition:** Determine one or more scenarios, events, or conditions that trigger the required behavior. This may also be empty.
3. **Identify the Required Action:** Specify one or more mandatory action or behavior required by the entity, as outlined in the SR. This must not be empty.
4. **Structure the Response:** Clearly separate the Trigger Condition and Required Action under distinct headings.

# Output Format

Present the information as follows:

- **Trigger Condition:** [Description of the condition(s), which may be empty]
- **Required Action:** [Description of the action]

# Examples

Example:

Specification Requirement:
A server MUST reject, with a response status code of 400 (Bad Request), any received request message that contains whitespace between a header field name and colon.

- **Trigger Condition:** A request message containing whitespace between a header field name and colon.
- **Required Action:** The server MUST reject the request and respond with a status code of 400 (Bad Request).

# Notes

- Ensure the extracted components are precise and directly supported by the SR.
- Context paragraphs are not required for extraction unless they significantly inform the SR interpretation.
"""

    prompt = f"""
    Here are the SR and its context:
      SR: {sr}
      Context: {context}"""

    try:
        completion = await client.beta.chat.completions.parse(
            model="o3-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format=RoleJudgmentResponse
        )

        match_reasoning = completion.choices[0].message.parsed
        actions = match_reasoning.actionlist
        conditions = match_reasoning.conditionlist
        return conditions, actions
        
    except Exception as e:
        print(f"Error calling API for SR: {sr[:30]}...: {str(e)}")
        # Return an empty list on error
        return [], []

# Function to create context from CSV fields
def create_context_from_csv(current_para, prev_para, next_para):
    context = ""
    if prev_para:
        context += "Previous paragraph:\n" + prev_para + "\n\n"
    
    context += "Current paragraph:\n" + current_para + "\n"
    
    if next_para:
        context += "\nNext paragraph:\n" + next_para
    
    return context

# Process a batch of rows asynchronously
async def process_batch(batch_rows):
    tasks = []
    for row in batch_rows:
        context = create_context_from_csv(
            row.get('Current Paragraph', ''),
            row.get('Previous Paragraph', ''),
            row.get('Next Paragraph', '')
        )
        tasks.append(call_gpt_api_async(row['SR Text'], context))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Update rows with results
    for i, row in enumerate(batch_rows):
        conditions, actions = results[i]
        row['Conditions'] = ','.join(conditions)
        row['Actions'] = ','.join(actions)
    
    return batch_rows, results

def should_process_sr(rfc_id: str, row: Dict[str, Any]) -> bool:
    """Check if the SR should be processed based on RFC ID and role."""
    # For TLS RFC, process all SRs
    if rfc_id == "8446":
        return True
    
    # For HTTP RFCs, only process SRs with specific roles
    if rfc_id in ["9110", "9111", "9112"]:
        role = row.get('Role', '').lower()
        target_roles = ['server', 'sender', 'recipient']
        return any(r in role for r in target_roles)
    
    return True

# Main function to process CSV asynchronously
async def process_csv_async(csv_file, output_json_file, output_csv_file, rfc_id, batch_size=10):
    start_time = time.time()
    print(f"Starting CSV processing with batch size {batch_size}...")
    
    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    # Filter rows based on RFC requirements
    rows = [row for row in all_rows if should_process_sr(rfc_id, row)]
    
    total_rows = len(rows)
    print(f"Total rows to process after filtering: {total_rows}")
    
    # Process in batches
    results = []
    for i in range(0, total_rows, batch_size):
        batch = rows[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size}")
        
        processed_batch, batch_results = await process_batch(batch)
        
        # Create result dictionaries for JSON output
        for j, row in enumerate(processed_batch):
            conditions, actions = batch_results[j]
            result = {
                'sr': row['SR Text'],
                'conditions': conditions,
                'actions': actions
            }
            
            # Add all other fields from CSV to the result
            for key, value in row.items():
                if key not in ['SR Text', 'Conditions', 'Actions']:
                    result[key] = value
            
            results.append(result)
        
        # Log progress
        processed_so_far = min(i + batch_size, total_rows)
        elapsed = time.time() - start_time
        rows_per_second = processed_so_far / elapsed if elapsed > 0 else 0
        print(f"Processed {processed_so_far}/{total_rows} rows. Speed: {rows_per_second:.2f} rows/sec")
    
    # Write results to JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Write results to CSV file
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as f:
        # Create new field order with Conditions and Actions after SR Text
        fieldnames = list(rows[0].keys())
        sr_index = fieldnames.index('SR Text')
        new_fieldnames = (
            fieldnames[:sr_index+1] + 
            ['Conditions', 'Actions'] + 
            [f for f in fieldnames if f not in ['SR Text', 'Conditions', 'Actions']]
        )
        
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    total_time = time.time() - start_time
    print(f"CSV processing completed in {total_time:.2f} seconds. Average speed: {total_rows/total_time:.2f} rows/sec")
    return results

async def main():
    # List of RFC IDs to process
    RFC_IDS = ["9110", "9111", "9112", "8446"]
    
    RFC_ID = "8446"
    print(f"\nProcessing RFC {RFC_ID}...")
    if RFC_ID == "8446":
        csv_input = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr.csv'
        csv_output = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_constraints.csv'
        jsonl_output_from_csv = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_constraints.jsonl'
    else:
        csv_input = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_role.csv'
        csv_output = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_constraints.csv'
        jsonl_output_from_csv = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_constraints.jsonl'
    
    # Batch size controls concurrency level
    batch_size = 90
    
    print("Processing CSV input...")
    await process_csv_async(csv_input, jsonl_output_from_csv, csv_output, RFC_ID, batch_size)
    print(f"Processing completed for RFC {RFC_ID}!")

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())