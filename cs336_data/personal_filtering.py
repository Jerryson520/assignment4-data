from __future__ import annotations

from itertools import islice

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import random
from typing import Any
import re
from tests.adapters import (
    run_extract_text_from_html_bytes,
    run_mask_emails,
    run_mask_phone_numbers,
    run_mask_ips
)

FAST_MODEL = fasttext.load_model("models/lid.176.bin")

def run_samples(warc_path: str, n: int=20):
    with (
        open(warc_path, "rb") as f_warc,
        open("results/personal_filtering_samples.txt", "w") as out
    ):
        records = []
        i = 1
        for record in ArchiveIterator(f_warc, record_types=WarcRecordType.response):
            url = record.headers.get("WARC-Target-URI")
            text = run_extract_text_from_html_bytes(record.reader.read())
            if not text or not text.strip():
                continue 
            masked, n_email = run_mask_emails(text)
            masked, n_phone = run_mask_phone_numbers(masked)
            masked, n_ip    = run_mask_ips(masked)
            
            if n_email + n_phone + n_ip == 0:
                continue
            else:
                out.write(f"\n{'='*30} 第 {i} 条 {'='*30}\n")
                out.write(f"URL: {url}\n")
                out.write(f"替换数: email={n_email} phone={n_phone} ip={n_ip}\n")
                out.write(f"\n----- 替换前 -----\n{text}\n")
                out.write(f"\n----- 替换后 -----\n{masked}\n")
                records.append((url, text))
                i += 1
            if i > n:
                break
            

if __name__ == "__main__":
    warc_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    run_samples(warc_path, 20)
