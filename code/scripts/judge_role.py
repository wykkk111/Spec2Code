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
    rolelist: List[str]

# Define async GPT API call function
async def call_gpt_api_async(sr, context):
    prompt = f"""
    You are a domain expert in the HTTP protocol RFC.
    This is a jsonl file that contains SR (specific requirements) from the HTTP protocol RFC and the corresponding chapters. 
    In the RFC, requirements are placed on various roles like senders, recipients, clients, servers, user agents, intermediaries, origin servers, proxies, gateways, or caches, depending on which behavior is constrained by the requirement. 
    I need you to analyze the following SR and its related context, then determine which role it applies to (e.g., sender, recipient, client, server, etc.).

    Here are the SR and its context:
      SR: {sr}
      Context: {context}
    """
    try:
        completion = await client.beta.chat.completions.parse(
            model="o3-mini",
            messages=[
                {"role": "system", "content": """ You are asked to analyze the specific requirements (SRs) from the RFC, then determine which role the SR is a constraint and specification for. The roles include [
  "Client",
  "Server",
  "Proxy",
  "Gateway",
  "Cache",
  "Sender",
  "Recipient",
  "Origin Server"
].
Think step by step.
"""},
                {"role": "user", "content": prompt}
            ],
            response_format=RoleJudgmentResponse
      )

        match_reasoning = completion.choices[0].message.parsed
        role = match_reasoning.rolelist
        return role
    except Exception as e:
        print(f"Error calling API for SR: {sr[:30]}...: {str(e)}")
        # Return an empty list on error
        return []

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
    roles = await asyncio.gather(*tasks)
    
    # Update rows with results
    for i, row in enumerate(batch_rows):
        row['Role'] = ','.join(roles[i])
    
    return batch_rows, roles

# Main function to process CSV asynchronously
async def process_csv_async(csv_file, output_json_file, output_csv_file, batch_size=10):
    start_time = time.time()
    print(f"Starting CSV processing with batch size {batch_size}...")
    
    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"Total rows to process: {total_rows}")
    
    # Process in batches
    results = []
    for i in range(0, total_rows, batch_size):
        batch = rows[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size}")
        
        processed_batch, roles = await process_batch(batch)
        
        # Create result dictionaries for JSON output
        for j, row in enumerate(processed_batch):
            result = {
                'sr': row['SR Text'],
                'role': roles[j]
            }
            
            # Add all other fields from CSV to the result
            for key, value in row.items():
                if key != 'SR Text':
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
    
    # Write results to CSV file with Role field after SR Text
    with open(output_csv_file, 'w', encoding='utf-8', newline='') as f:
        # Create new field order with Role after SR Text
        fieldnames = list(rows[0].keys())
        sr_index = fieldnames.index('SR Text')
        new_fieldnames = fieldnames[:sr_index+1] + ['Role'] + [f for f in fieldnames if f != 'SR Text' and f != 'Role']
        
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    total_time = time.time() - start_time
    print(f"CSV processing completed in {total_time:.2f} seconds. Average speed: {total_rows/total_time:.2f} rows/sec")
    return results

# Main entry point that runs both processes asynchronously
async def main():
    RFC_ID = '9110'  # Change this to the RFC ID you are working with
    # Process CSV file
    csv_input = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr.csv'
    csv_output = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_role.csv'
    jsonl_output_from_csv = f'/data/a/ykw/RFC/final/sr/{RFC_ID}sr_role.jsonl'
    
    # Batch size controls concurrency level
    batch_size = 104  # Adjust based on API rate limits and system capabilities
    
    if os.path.exists(csv_input):
        print("Processing CSV input...")
        results = await process_csv_async(csv_input, jsonl_output_from_csv, csv_output, batch_size)
        print("All processing completed!")
    else:
        print("CSV input file not found!")

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())