"""Reduced-scale local WET download for self-study (no Modal, no course shared volume).

Downloads n_files WET files from Common Crawl with curl (retries + resume),
filters records to English (lid.176.bin, probability >= 0.7), and writes
filtered chunk files to local-shared-data/english-wet-data/.

Resumable: already-written chunk outputs are skipped on rerun.

Usage: uv run python scripts/download_wet_local.py --n-files 100
"""

import argparse
import gzip
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import fasttext
import polars as pl
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.common import get_shared_assets_path

BASE_URL = "https://data.commoncrawl.org/"
CRAWL_ID = "CC-MAIN-2026-17"
GROUP_SIZE = 4
SHUFFLE_SEED = 336


def curl(url: str, out: Path) -> None:
    subprocess.run(
        ["curl", "-sL", "--retry", "10", "--retry-all-errors", "-C", "-", "-o", str(out), url],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=100, help="Number of raw WET files (course full scale is 2500)")
    args = parser.parse_args()
    assert args.n_files % GROUP_SIZE == 0

    shared = get_shared_assets_path()
    out_dir = shared / "english-wet-data"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.gettempdir()) / "cs336-wet"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1. WET path listing (same sampling as EnglishWetFiles for faithfulness)
    paths_gz = tmp_dir / "wet.paths.gz"
    if not paths_gz.exists():
        print("downloading wet.paths.gz")
        curl(f"{BASE_URL}crawl-data/{CRAWL_ID}/wet.paths.gz", paths_gz)
    wet_paths = pl.read_csv(paths_gz, has_header=False, new_columns=["wet_path"]).sample(
        n=args.n_files, shuffle=True, seed=SHUFFLE_SEED, with_replacement=False
    )["wet_path"]
    wet_urls = [BASE_URL + p for p in wet_paths]

    lid_model = fasttext.load_model(str(shared / "classifiers" / "lid.176.bin"))

    def is_english(text: str) -> bool:
        labels, probs = lid_model.predict(text.replace("\n", " "))
        return labels[0] == "__label__en" and probs[0] >= 0.7

    # 2. Download + filter, one chunk (GROUP_SIZE raw files) per output file
    n_chunks = len(wet_urls) // GROUP_SIZE
    for chunk_idx in range(n_chunks):
        out_path = out_dir / f"{chunk_idx:05d}-data.warc.wet.gz"
        if out_path.exists():
            print(f"[{chunk_idx + 1}/{n_chunks}] exists, skipping")
            continue
        chunk_urls = wet_urls[chunk_idx * GROUP_SIZE : (chunk_idx + 1) * GROUP_SIZE]
        total, kept = 0, 0
        tmp_out = out_path.with_suffix(".tmp")
        with gzip.open(tmp_out, "wb") as output_stream:
            writer = WARCWriter(output_stream, gzip=False)
            for wet_url in chunk_urls:
                local_wet = tmp_dir / wet_url.split("/")[-1]
                print(f"[{chunk_idx + 1}/{n_chunks}] downloading {local_wet.name}")
                curl(wet_url, local_wet)
                with gzip.open(local_wet, "rb") as input_stream:
                    for rec in ArchiveIterator(input_stream):
                        if rec.rec_type != "conversion":
                            writer.write_record(rec)
                            continue
                        payload = rec.content_stream().read()
                        total += 1
                        if is_english(payload.decode("utf-8", errors="replace")):
                            kept += 1
                            rec.raw_stream = BytesIO(payload)
                            writer.write_record(rec)
                local_wet.unlink()  # free disk as we go
        tmp_out.rename(out_path)
        print(f"[{chunk_idx + 1}/{n_chunks}] wrote {out_path.name}: kept {kept}/{total} records")

    print(f"Done: {n_chunks} filtered English WET chunks in {out_dir}")


if __name__ == "__main__":
    main()
