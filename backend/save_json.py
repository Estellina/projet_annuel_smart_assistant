import json
from datetime import datetime

def save_results(data, output_dir="data"):
    filename = f"{output_dir}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
