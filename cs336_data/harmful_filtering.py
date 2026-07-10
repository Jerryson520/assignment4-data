from __future__ import annotations

import os
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
from tests.adapters import (
    run_classify_nsfw,
    run_classify_toxic_speech,
    run_extract_text_from_html_bytes
)
from itertools import islice
import random

def run_samples(warc_path: str, n: int=20, pool: int=500, seed:int=42):
    with open(warc_path, "rb") as f_warc:
        records = []
        for record in islice(ArchiveIterator(f_warc, record_types=WarcRecordType.response), pool):
            url = record.headers.get("WARC-Target-URI")
            text = run_extract_text_from_html_bytes(record.reader.read())
            if text and text.strip():
                records.append((url, text))
        random.seed(seed)
        samples = random.sample(records, n)

    with open("results/harmful_samples.txt", "w") as out:
        for i, (url, text) in enumerate(samples, 1):
            nsfw_label, nsfw_score = run_classify_nsfw(text)
            toxic_label, toxic_score = run_classify_toxic_speech(text)  
            out.write(f"\n{'='*30} 第 {i} 条 {'='*30}\n")
            out.write(f"URL: {url}\n")
            out.write(f"NSFW: {nsfw_label} ({nsfw_score:.3f})  毒性: {toxic_label} ({toxic_score:.3f})\n")
            out.write(text + "\n")   
            

if __name__ == "__main__":
    warc_path = "local-shared-data/CC/example.warc.gz"
    run_samples(warc_path, 20)
