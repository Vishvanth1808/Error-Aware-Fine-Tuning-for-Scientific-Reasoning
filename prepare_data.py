import json

INPUT_FILE = "dataset.jsonl"
OUTPUT_FILE = "formatted_dataset.jsonl"

def format_example(ex):
    return {
        "text": f"""
You are a scientific reasoning assistant that corrects student mistakes.

Problem:
{ex["instruction"]}

Student Answer:
{ex["input"]}

Error Type:
{ex["error_type"]}

Error Explanation:
{ex["error_explanation"]}

Correction Strategy:
{ex["correction_strategy"]}

Correct Explanation:
{ex["output"]}
"""
    }

with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
    for line in fin:
        ex = json.loads(line)
        formatted = format_example(ex)
        fout.write(json.dumps(formatted) + "\n")

print("formatted_dataset.jsonl generated successfully.")