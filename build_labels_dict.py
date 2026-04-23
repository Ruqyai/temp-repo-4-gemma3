"""
Build a dictionary mapping feature indices to their labels
from delphi explanation files.

Usage:
    python build_labels_dict.py \
        --explanations_dir results/gemma3_4b_it_layer15/explanations \
        --output feature_labels.json
"""

import json
import re
from pathlib import Path
import argparse


def build_labels_dict(explanations_dir: str) -> dict[int, str]:
    """
    Read all explanation .txt files and build {feature_index: label} dict.
    
    File naming convention:
        language_model.layers.15.mlp_latent{INDEX}.txt
    """
    labels = {}
    explanations_path = Path(explanations_dir)

    for txt_file in sorted(explanations_path.glob("*.txt")):
        # Extract feature index from filename
        match = re.search(r"_latent(\d+)\.txt$", txt_file.name)
        if match:
            feature_index = int(match.group(1))
            label = txt_file.read_text().strip().strip('"')
            labels[feature_index] = label

    return labels


def main():
    parser = argparse.ArgumentParser(description="Build feature labels dictionary")
    parser.add_argument(
        "--explanations_dir",
        type=str,
        default="results/gemma3_4b_it_layer15/explanations",
        help="Path to the explanations directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="feature_labels.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Build the dict
    labels = build_labels_dict(args.explanations_dir)

    # Save to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Total features labeled: {len(labels)}")
    print(f"Saved to: {args.output}")

    # Print a few examples
    print("\nExamples:")
    for idx, (k, v) in enumerate(sorted(labels.items())[:5]):
        print(f"  Feature {k}: {v[:80]}...")


if __name__ == "__main__":
    main()
