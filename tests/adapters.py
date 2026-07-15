from __future__ import annotations

import os
from typing import Any

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import re
from nltk import word_tokenize
from collections import Counter
import mmh3
import random
import unicodedata
from pathlib import Path
from collections import defaultdict
from itertools import combinations

LANG_MODEL = fasttext.load_model("models/lid.176.bin")
NSFW_MODEL = fasttext.load_model("models/jigsaw_fasttext_bigrams_nsfw_final.bin")
HATE_SPEECH_MODEL = fasttext.load_model("models/jigsaw_fasttext_bigrams_hatespeech_final.bin")
QUALITY_MODEL = fasttext.load_model("models/quality_classifier.bin")

def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding, errors="ignore")
    text = extract_plain_text(html_str)
    return text


def run_identify_language(text: str) -> tuple[Any, float]:
    model = LANG_MODEL
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
    model = NSFW_MODEL
    label, scores = model.predict(text.replace("\n", " ").strip())
    label = label[0][9:]
    return label, scores[0]


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = HATE_SPEECH_MODEL
    label, scores = model.predict(text.replace("\n", " ").strip())
    label = label[0][9:]
    return label, scores[0]

def run_classify_quality(text: str) -> tuple[Any, float]:
    model = QUALITY_MODEL
    label, scores = model.predict(text.replace("\n", " ").strip())
    label = label[0][9:]
    return label, scores[0]


def run_gopher_quality_filter(text: str) -> bool:
    words = word_tokenize(text)
    lines = text.split("\n")
    word_len = len(words)

    if word_len < 50 or word_len > 100000:
        return False

    total_word_len = sum(len(word) for word in words)
    if total_word_len / word_len < 3 or total_word_len / word_len > 10:
        return False

    ellipsis_count =sum(line.endswith("...") for line in lines)
    if ellipsis_count / len(lines) > 0.3:
        return False

    def has_alpha(word):
        return any(chr.isalpha() for chr in word)

    total_alpha = sum(has_alpha(word) for word in words)
    if total_alpha / word_len < 0.8:
        return False

    return True

def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    input_files = [Path(path) for path in input_files]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    line_counts = Counter()
    for file in input_files:
        with file.open("rb") as f:
            for line in f:
                h = mmh3.hash128(line, seed=42)
                line_counts[h] += 1

    for input_path in input_files:
        output_path = output_directory / input_path.name
        with (
            input_path.open("rb") as f,
            output_path.open("wb") as f_output
        ):
            for line in f:
                h = mmh3.hash128(line, seed=42)
                if line_counts[h] > 1:
                    continue
                f_output.write(line)

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    text = " ".join(text.split())
    return text

def get_ngrams(text: str, ngrams: int) -> set[tuple[str, ...]]:
    text = normalize_text(text)
    grams = text.split()
    gram_set = {
        tuple(grams[i: i+ngrams])
        for i in range(len(grams) - ngrams + 1)
    }
    return gram_set

def encode_hash(ngram: tuple[str, ...], seed: int) -> int:
    encoded = "\x1f".join(ngram)

    return mmh3.hash128(encoded, seed, signed=False)

def jaccard_similarity(
    left: set[tuple[str, ...]],
    right: set[tuple[str, ...]]
) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")

    if num_bands <= 0:
        raise ValueError("num_bands must be positive")

    if ngrams <= 0:
        raise ValueError("ngrams must be positive")

    if num_hashes % num_bands != 0:
        raise ValueError(
            "num_hashes must be divisible by num_bands"
        )

    rows_per_band = num_hashes // num_bands

    rng = random.Random(336)
    seeds = rng.sample(
        range(2**32),
        k=num_hashes,
    )

    input_files = [Path(input_file) for input_file in input_files]
    output_directory = Path(output_directory)

    signatures = []
    document_ngrams = []
    for file in input_files:
        doc = file.read_text(encoding="utf-8", errors="replace")
        gram_set = get_ngrams(doc, ngrams)
        if gram_set:
            signature = tuple(
                min(encode_hash(gram, seed) for gram in gram_set)
                for seed in seeds
            )
        else:
            signature = tuple(2**128 - 1 for _ in seeds)

        signatures.append(signature)
        document_ngrams.append(gram_set)

    buckets: dict[
        tuple[int, tuple[int, ...]],
        list[int]
    ] = defaultdict(list)
    for document_idx, sig in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            key = (band_idx, sig[start:end])
            buckets[key].append(document_idx)

    pairs = set()
    for val in buckets.values():
        if len(val) > 1:
            pairs.update(combinations(val, 2))

    duplicate_pairs: list[tuple[int, int]] = []
    for left, right in pairs:
        score = jaccard_similarity(
                document_ngrams[left],
                document_ngrams[right],
            )
        if score >= jaccard_threshold:
            duplicate_pairs.append((left, right))

    parent = list(range(len(input_files)))
    def find(document_idx: int):
        if parent[document_idx] != document_idx:
            parent[document_idx] = find(parent[document_idx])

        return parent[document_idx]

    def union(left: int, right: int):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for left, right in duplicate_pairs:
        union(left, right)

    groups = defaultdict(list)
    for document_idx in range(len(input_files)):
        root = find(document_idx)
        groups[root].append(document_idx)

    kept_indices = sorted(min(group) for group in groups.values())

    output_directory.mkdir(parents=True, exist_ok=True)
    for idx in kept_indices:
        input_path = input_files[idx]
        output_file = output_directory / input_path.name

        output_file.write_bytes(input_path.read_bytes())













if __name__ == "__main__":
    print(run_identify_language("欢迎来到我们的网站"))
