from __future__ import annotations

import json
import random
from pathlib import Path


DATA_DIRECTORY = Path("local-shared-data/filtered-data/final")
NUM_SAMPLES = 5
SEED = 336

def main():
    input_paths = sorted(DATA_DIRECTORY.glob("*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"No JSONL files found in {DATA_DIRECTORY}")

    rng = random.Random(SEED)
    samples: list[dict] = []
    documents_seen = 0

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                document = json.loads(line)
                documents_seen += 1

                if len(samples) < NUM_SAMPLES:
                    samples.append(document)
                    continue

                replace_idx = rng.randrange(documents_seen)
                if replace_idx < NUM_SAMPLES:
                    samples[replace_idx] = document
                
    print(f"Sampled {NUM_SAMPLES} of {documents_seen} documents")

    for index, document in enumerate(samples, start=1):
        excerpt = " ".join(
            document["text"].split()
        )

        print()
        print("=" * 80)
        print(f"Sample {index}")
        print(f"URL: {document['url']}")
        print(f"Domain: {document['domain']}")
        print(f"Word count: {document['word_count']}")
        print(f"Excerpt: {excerpt}")

if __name__ == "__main__":
    main()