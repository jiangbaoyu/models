import json
from pathlib import Path

for file in ["data/train.json", "data/dev.json"]:
    print(f"\n=== {file} ===")
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(json.loads(line))