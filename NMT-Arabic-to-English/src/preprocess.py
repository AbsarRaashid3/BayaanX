# src/preprocess.py

import argparse
from datasets import load_dataset

def preprocess_and_save(output_file, split):
    # Load the dataset using the provided split ("test" or "validation")
    dataset = load_dataset('Helsinki-NLP/tatoeba_mt', 'ara-eng', split=split, trust_remote_code=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in dataset:
            # Use the appropriate keys from your dataset
            ara = ex['sourceString'].strip().lower()
            eng = ex['targetString'].strip().lower()
            if ara and eng:
                f.write(f"{ara}\t{eng}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True, help="Output file for preprocessed pairs")
    parser.add_argument('--split', type=str, default="test", help='Which split to use: "test" or "validation"')
    args = parser.parse_args()
    preprocess_and_save(args.output_file, args.split)
