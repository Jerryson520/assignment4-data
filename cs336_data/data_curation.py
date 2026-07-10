from __future__ import annotations

from itertools import islice

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import random
from typing import Any

FAST_MODEL = fasttext.load_model("models/lid.176.bin")

def run_identify_language(text: str) -> tuple[Any, float]:
    model = FAST_MODEL
    label, scores = model.predict(text.replace("\n", " ").strip())
    label = label[0][9:]
    return label, scores[0]

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors="ignore")
    text = extract_plain_text(html_str)
    return text


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

    with open("results/langid_samples.txt", "w") as out:
        for i, (url, text) in enumerate(samples, 1):
            lang, score = run_identify_language(text)
            out.write(f"\n{'='*30} 第 {i} 条 {'='*30}\n")
            out.write(f"URL: {url}\n预测: {lang}  置信度: {score:.3f}\n")
            out.write(f"我的人工标注: ______\n")   # ← 留空自己填
            out.write(text[:800] + "\n")           # 正文预览，够人工判断即可

if __name__ == "__main__":
    warc_path = "local-shared-data/CC/example.warc.gz"
    run_samples(warc_path, 20, 500)
