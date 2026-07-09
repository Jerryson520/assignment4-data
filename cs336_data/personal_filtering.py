from __future__ import annotations

from itertools import islice

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import random
from typing import Any
import re

FAST_MODEL = fasttext.load_model("models/lid.176.bin")

def run_mask_emails(text: str) -> tuple[str, int]:
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    new_string, count = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return new_string, count

def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r"(?<!\d)\(?\d{3}\)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{4}(?!\d)"
    new_string, count = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return new_string, count

def run_mask_ips(text: str) -> tuple[str, int]:
    pattern = r"(?<!\d)((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
    new_string, count = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return new_string, count

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors="ignore")
    text = extract_plain_text(html_str)
    return text


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
