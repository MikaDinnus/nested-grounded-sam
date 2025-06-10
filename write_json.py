import os
import json

def save_json(data: dict):
    dataset_number = data.get("CODE", "unknown")
    type_ = data.get("TYPE", "unknown")
    output_dir = os.path.join(f"evaluation_images/{dataset_number}_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{dataset_number}_{type_}_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"JSON gespeichert unter: {output_path}")
