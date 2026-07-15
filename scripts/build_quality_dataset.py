import random
import re
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from tests.adapters import (
    run_extract_text_from_html_bytes,
    run_gopher_quality_filter,
    run_identify_language,
    run_classify_nsfw,
    run_classify_toxic_speech
)

WIKI_WARC = Path("quality_data/wiki_positive.warc.gz")
CC_WARC = Path("local-shared-data/CC/example.warc.gz")

TRAIN_PATH = Path("quality_data/quality.train")
VALID_PATH = Path("quality_data/quality.valid")

SEED = 336
MAX_PER_CLASS = 3000


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_documents(path: Path, apply_quality_filter: bool) -> list[str]:
    documents = []
    stats = Counter()

    with open(path, "rb") as f:
        records = ArchiveIterator(
            f,
            WarcRecordType.response,
        )

        progress = tqdm(records, desc=f"Reading {path.name}", unit="page")
        for record in progress:
            stats["seen"] += 1
            try:
                record.parse_http()
                if record.http_headers is None:
                    stats["no_http_headers"] += 1
                    continue
                status = record.http_headers.status_code
                if status != 200:
                    stats[f"http_{status}"] += 1
                    continue

                content_type = record.http_headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    stats["non_html"] += 1
                    continue

                html_bytes = record.reader.read()
                text = run_extract_text_from_html_bytes(html_bytes)
                if not text:
                    stats["empty"] += 1
                    continue

                text = normalize(text)

                lang, score = run_identify_language(text)
                if lang != "en" or score < 0.7:
                    stats["non_english"] += 1
                    continue

                if apply_quality_filter:
                    if not run_gopher_quality_filter(text):
                        stats["gopher_rejected"] += 1
                        continue

                    label, score = run_classify_nsfw(text)
                    if label == "nsfw" or (label == "non-nsfw" and score < 0.7):
                        stats["nsfw_rejected"] += 1
                        continue
                    label, score = run_classify_toxic_speech(text)
                    if label == "toxic" or (label == "non-toxic" and score < 0.7):
                        stats["toxic_rejected"] += 1
                        continue

                documents.append(text)
                stats["kept"] += 1

            except Exception as error:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    tqdm.write(f"{path.name}: {type(error).__name__}: {error}")
            finally:
                if stats["seen"] % 100 == 0:
                    progress.set_postfix(
                        kept=stats["kept"],
                        non_en=stats["non_english"],
                        rejected=stats["gopher_rejected"]
                        + stats["nsfw_rejected"]
                        + stats["toxic_rejected"],
                        errors=stats["errors"],
                        refresh=True,
                    )

    tqdm.write(f"Finished {path.name}: {dict(stats)}")
    return documents

wiki_documents = extract_documents(WIKI_WARC, apply_quality_filter=True)
cc_documents = extract_documents(CC_WARC, apply_quality_filter=False)

rng = random.Random(SEED)
rng.shuffle(wiki_documents)
rng.shuffle(cc_documents)


n = min(MAX_PER_CLASS, len(wiki_documents), len(cc_documents))
wiki_documents = wiki_documents[:n]
cc_documents = cc_documents[:n]

examples = [("wiki", text) for text in wiki_documents] + [("cc", text) for text in cc_documents]

rng.shuffle(examples)

split = int(len(examples) * 0.9)
train_examples = examples[:split]
valid_examples = examples[split:]

def write_fasttext(path: Path, samples: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for label, text in tqdm(
            samples,
            desc=f"Writing {path.name}",
            unit="example",
        ):
            f.write(f"__label__{label} {text}\n")

write_fasttext(TRAIN_PATH, train_examples)
write_fasttext(VALID_PATH, valid_examples)

print(f"wiki candidates: {len(wiki_documents)}")
print(f"cc candidates:   {len(cc_documents)}")
print(f"train examples:  {len(train_examples)}")
print(f"valid examples:  {len(valid_examples)}")
