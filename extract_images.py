import json
from pathlib import Path

log_path = Path(r"C:\Users\Anto SJ\.gemini\antigravity\brain\a898b680-b16b-4a75-9e75-cb33ce7ed1e3\.system_generated\logs\transcript_full.jsonl")

if not log_path.exists():
    print("Log path not found!")
    exit(1)

with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find all user input steps
user_inputs = []
for line in lines:
    try:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            user_inputs.append(data)
    except Exception as e:
        continue

if not user_inputs:
    print("No user input steps found in log!")
    exit(1)

# Get the latest user input
latest_input = user_inputs[-1]
print("Keys in user input:", latest_input.keys())

# Let's inspect the content structure of the user input
content = latest_input.get("content", "")
print("Content length:", len(content))

# Print a snippet of content to see if it is JSON or text
print("Content snippet:", content[:500])

# Check if there are other keys like files, images, etc.
for k, v in latest_input.items():
    if k not in ["content", "type", "step_index", "source", "status"]:
        print(f"Key: {k}, type: {type(v)}, value snippet: {str(v)[:300]}")
