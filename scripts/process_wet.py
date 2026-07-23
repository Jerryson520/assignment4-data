from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from tests.adapters import (
    run_classify_nsfw,
    run_classify_toxic_speech,
    run_classify_quality,
    run_gopher_quality_filter,
    run_mask_emails,
    run_mask_phone_numbers,
    run_mask_ips
)
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
import time
from collections import Counter
import json

NUM_OF_CPUS = 2
ROOT_DIR = Path("local-shared-data/english-wet-data")
EXP_NAME = "final"
OUTPUT_DIRECTORY = Path("local-shared-data/filtered-data/" + EXP_NAME)
DOMAIN_EXTRACTOR = TLDExtract(suffix_list_urls=())

def iter_wet_files(input_path: str):
    with open(input_path, "rb") as f:
        records = ArchiveIterator(
            f,
            record_types = WarcRecordType.conversion,
            parse_http = False
        )

        for record in records:
            text = record.reader.read().decode("utf-8", errors="replace")

            yield {
                "url": record.headers.get("WARC-Target-URI") or "",
                "date": record.headers.get("WARC-Date") or "",
                "record_id": record.headers.get("WARC-Record-ID") or "",
                "text": text,
            }

def extract_domain(url: str) -> str:
    if not url:
        return ""
    extracted_url = DOMAIN_EXTRACTOR(url)
    return extracted_url.top_domain_under_public_suffix


def get_rejection_reason(text: str) -> str | None:
    text = text.strip()
    if not text:
        return "empty"
    
    word_count = len(text.split())
    if word_count < 50:
        return "too_short"
    
    if word_count > 100000:
        return "too_long"

    if not run_gopher_quality_filter(text):
        return "gopher"
    
    if run_classify_quality(text)[0] == "cc":
        return "low_quality"

    if run_classify_nsfw(text)[0] == "nsfw":
        return "nsfw"
    
    if run_classify_toxic_speech(text)[0] == "toxic":
        return "toxic"

    return None


def process_single_file(
    input_path: str, 
    output_path: str
) -> dict[str, int | float | str]:
    started_at = time.perf_counter()
    stats = Counter()
    output_path = Path(output_path)
    temp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with temp_output_path.open("w", encoding="utf-8") as output_file:
        for document in iter_wet_files(input_path):
            stats["total"] += 1

            reason = get_rejection_reason(document["text"])

            if reason is not None:
                stats[f"rejected_{reason}"] += 1
                continue

            text = document["text"].strip()
            text, email_count = run_mask_emails(text)
            text, phone_count = run_mask_phone_numbers(text)
            text, ip_count = run_mask_ips(text)

            stats["masked_emails"] += email_count
            stats["masked_phone_numbers"] += phone_count
            stats["masked_ips"] += ip_count

            if (email_count + phone_count + ip_count) > 0:
                stats["modified_pii_documents"] += 1

            output_record = {
                "url": document["url"],
                "domain": extract_domain(document["url"]),
                "date": document["date"],
                "record_id": document["record_id"],
                "word_count": len(text.split()),
                "text": text
            }

            output_file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )

            stats["kept"] += 1
    
    temp_output_path.replace(output_path)
    elapsed_seconds = (
        time.perf_counter() - started_at
    )

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "elapsed_seconds": elapsed_seconds,
        **stats
    }

def main():
    wet_files = sorted(ROOT_DIR.glob("*.warc.wet.gz"))

    if not wet_files:
        raise FileNotFoundError(
            f"No WET files Found in {ROOT_DIR}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = time.perf_counter()

    per_file_stats = []
    totals = Counter()


    with ProcessPoolExecutor(max_workers=NUM_OF_CPUS) as executor:
        future_to_input = {}

        for wet_path in wet_files:
            output_path = OUTPUT_DIRECTORY / f"{Path(wet_path).name.removesuffix('.warc.wet.gz')}.jsonl"
            future = executor.submit(
                process_single_file,
                str(wet_path),
                str(output_path)
            )

            future_to_input[future] = wet_path

        for future in tqdm(
            as_completed(future_to_input), 
            total=len(future_to_input),
            desc="Processing WET files"
        ): 
            input_path = future_to_input[future]

            try:
                stats = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Failed to process {input_path}"
                ) from error
            
            per_file_stats.append(stats)

            for key in (
                "total",
                "kept",
                "rejected_empty",
                "rejected_too_short",
                "rejected_too_long",
                "rejected_nsfw",
                "rejected_toxic",
                "rejected_low_quality",
                "rejected_gopher",
                "modified_pii_documents",
                "masked_emails",
                "masked_phone_numbers",
                "masked_ips",
            ):
                totals[key] += int(stats.get(key, 0))

    per_file_stats.sort(key=lambda item: item["input_path"])
    elapsed_seconds = (
        time.perf_counter() - started_at
    )
    total = totals["total"]
    kept = totals["kept"]

    report = {
        "experiment": EXP_NAME,
        "num_workers": NUM_OF_CPUS,
        "num_input_files": len(wet_files),
        "elapsed_seconds": elapsed_seconds,
        "totals": {
            **totals,
            "keep_ratio": (
                kept / total
                if total else 0.0
            ),
        },
        "per_file": per_file_stats,
    }

    report_path = (OUTPUT_DIRECTORY / "filter_report.json")

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()