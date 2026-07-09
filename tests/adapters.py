from __future__ import annotations

import os
from typing import Any
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding)
    text = extract_plain_text(html_str)
    return text


def run_identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("models/lid.176.bin")
    label, scores = model.predict(text.replace("\n", " ").strip())
    label = label[0][9:]
    return label, max(scores)

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


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    raise NotImplementedError


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError

if __name__ == "__main__":
    print(run_identify_language("欢迎来到我们的网站"))