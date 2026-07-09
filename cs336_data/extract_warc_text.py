from __future__ import annotations

from itertools import islice

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors="ignore")
    text = extract_plain_text(html_str)
    return text

def compare_warc_wet(warc_path: str, wet_path: str, n: int = 20):
    with (
        open(warc_path, "rb") as f_warc,
        open(wet_path, "rb") as f_wet,
        open("results/my_extract.txt", "w") as f_mine,     # 我的提取结果
        open("results/wet_extract.txt", "w") as f_wetout,  # WET 官方提取结果
    ):
        warc_iter = ArchiveIterator(f_warc, record_types=WarcRecordType.response)
        wet_iter = ArchiveIterator(f_wet, record_types=WarcRecordType.conversion)

        for i, (warc_rec, wet_rec) in enumerate(islice(zip(warc_iter, wet_iter), n), 1):
            url = warc_rec.headers.get("WARC-Target-URI")
            wet_url = wet_rec.headers.get("WARC-Target-URI")

            my_text = run_extract_text_from_html_bytes(warc_rec.reader.read())
            wet_text = wet_rec.reader.read().decode("utf-8")

            header = f"\n{'=' * 30} 第 {i} 条 {'=' * 30}\n"
            f_mine.write(header + f"URL: {url}\n\n" + my_text + "\n")
            f_wetout.write(header + f"URL: {wet_url}\n\n" + wet_text + "\n")


if __name__ == "__main__":
    warc_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    wet_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
    compare_warc_wet(warc_path, wet_path)
