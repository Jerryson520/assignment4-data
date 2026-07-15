import html
import random
from urllib.parse import urlsplit

INPUT = "local-shared-data/enwiki_extracted_urls.txt"
OUTPUT = "quality_data/wiki_urls.txt"
SEED = 336
NUM_URLS = 10000

def valid_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and "." in parsed.netloc
        )
    except ValueError:
        return False


with open(INPUT, encoding="utf-8") as f:
    urls = {
        html.unescape(line.strip())
        for line in f
        if line.strip()
    }

urls = [url for url in urls if valid_url(url)]

random.Random(SEED).shuffle(urls)
urls = urls[:NUM_URLS]

with open(OUTPUT, "w", encoding="utf-8") as f:
    for url in urls:
        f.write(url + "\n")

print(f"Wrote {len(urls)} URLs to {OUTPUT}")
