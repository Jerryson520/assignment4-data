from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool

EXP_NAME = "final"
INPUT_DIRECTORY = Path(f"local-shared-data/filtered-data/{EXP_NAME}")
OUTPUT_PATH = Path(f"local-shared-data/tokenized/{EXP_NAME}.bin")
TEMP_OUTPUT_PATH = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
REPORT_PATH = Path(f"local-shared-data/tokenized/{EXP_NAME}_tokenization_report.json")
BATCH_SIZE = 256
NUM_WORKERS = 4

def initialize_tokenizer():
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**30

def iter_text_batches(
    input_paths: list[Path],
    batch_size: int,
):
    batch = []

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                document = json.loads(line)
                batch.append(document["text"])

                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch

def tokenize_batch(texts: list[str]) -> list[list[int]]:
    return tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )["input_ids"]


def main():
    started_at = time.perf_counter()

    input_paths = sorted(INPUT_DIRECTORY.glob("*.jsonl"))
    if not input_paths:
        raise FileNotFoundError(
            f"No JSONL files found in {INPUT_DIRECTORY}"
        )

    eos_id = AutoTokenizer.from_pretrained("gpt2").eos_token_id

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    document_count = 0
    token_count = 0
    
    batches = iter_text_batches(input_paths, BATCH_SIZE)
    with (
        TEMP_OUTPUT_PATH.open("wb") as output_file,
        Pool(
            processes=NUM_WORKERS,
            initializer=initialize_tokenizer,
        ) as pool
    ):
        encoded_batches = pool.imap(
            tokenize_batch,
            batches,
            chunksize=1,
        )

        for encoded_batch in tqdm(encoded_batches, desc="Tokenizing"):
            for token_ids in encoded_batch:
                token_ids.append(eos_id)

                array = np.array(
                    token_ids,
                    dtype=np.uint16
                )

                array.tofile(output_file)

                document_count += 1
                token_count += len(token_ids)
    
    TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)
    elapsed_seconds = time.perf_counter() - started_at

    report = {
        "experiment": EXP_NAME,
        "num_workers": NUM_WORKERS,
        "batch_size": BATCH_SIZE,
        "input_files": len(input_paths),
        "documents": document_count,
        "tokens": token_count,
        "eos_token_id": eos_id,
        "dtype": "uint16",
        "elapsed_seconds": elapsed_seconds,
        "output_path": str(OUTPUT_PATH),
        "output_bytes": OUTPUT_PATH.stat().st_size,
    }


    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
