from __future__ import annotations

import json
import random
from pathlib import Path


from process_wet import (
    ROOT_DIR,
    get_rejection_reason,
    iter_wet_files,
)

NUM_CANDIDATES = 100
NUM_SAMPLES = 5
SEED = 336

def main():
    input_paths = sorted(ROOT_DIR.glob("*.warc.wet.gz"))
    if not input_paths:
        raise FileNotFoundError(f"No WET files found in {ROOT_DIR}")
    
    candidates = []
    documents_seen = 0
    rng = random.Random(SEED)
    
    for input_path in input_paths:
        for document in iter_wet_files(input_path):
            documents_seen += 1
            if len(candidates) < NUM_CANDIDATES:
                candidates.append(document)
                continue
            
            replacement_idx = rng.randrange(documents_seen)
            if replacement_idx < NUM_CANDIDATES:
                candidates[replacement_idx] = document

    rng.shuffle(candidates)
    rejected_samples = []
    for document in candidates:
        reason = get_rejection_reason(document["text"])
        if not reason:
            continue

        rejected_samples.append({**document, "rejection_reason": reason})

        if len(rejected_samples) == NUM_SAMPLES:
            break
    
    if len(rejected_samples) < NUM_SAMPLES:
        raise RuntimeError(
            "Not enough rejected samples; "
            "increase NUM_CANDIDATES."
        )

    print(
        f"Sampled {NUM_SAMPLES} rejected documents "
        f"from {documents_seen} original documents"
    )

    for index, document in enumerate(
        rejected_samples,
        start=1,
    ):
        print()
        print("=" * 80)
        print(f"Sample {index}")
        print(f"URL: {document['url']}")
        print(
            "Rejection reason: "
            f"{document['rejection_reason']}"
        )
        print(f"Text:\n{document['text']}")


if __name__ == "__main__":
    main()        