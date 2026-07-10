#!/usr/bin/env python3
"""本地版 wiki 外链抽取：下载 enwiki dump 分片 -> 抽 <ref> 里的 URL -> 删分片。
用法:
    uv run python scripts/extract_wiki_urls_local.py --n-shards 2
输出:
    local-shared-data/enwiki_extracted_urls.txt   (每行一个去重后的 URL)
"""
import argparse
import bz2
import json
import re
import subprocess
from pathlib import Path

DUMP_DATE = "20260501"
BASE = f"https://dumps.wikimedia.org/enwiki/{DUMP_DATE}"

# Wikimedia 会封杀默认的 "Python-urllib" UA（403），且 urllib 对大响应会 IncompleteRead 截断；
# 直接用 curl 更稳（自带重试 + 断点续传 + 失败非零退出）
UA = "cs336-selfstudy/1.0 (syu010520@gmail.com)"


def _retrieve(url, dest):
    subprocess.run(
        ["curl", "-fSL", "--retry", "5", "--retry-all-errors",
         "-A", UA, "-o", str(dest), url],
        check=True,
    )

# 和官方脚本一致的 URL 正则 + <ref> 匹配（dump 里尖括号被转义成 &lt; / &gt;）
URL_RE = re.compile(
    r"\b(?:https?|telnet|gopher|file|wais|ftp):"
    r"[\w/#~:.?+=&%@!\-.:?\\-]+?"
    r"(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
)
REF_RE = re.compile("&lt;ref&gt(.*)&lt;/ref&gt;")


def list_shards() -> list[str]:
    """从 dumpstatus.json 拿 pages-articles-multistream 分片文件名（按大小排序取小的先下）。"""
    # 直接 json.load(urlopen) 对大响应容易 IncompleteRead，先稳健落地成文件再解析
    status_path = Path("/tmp/wiki/dumpstatus.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    _retrieve(f"{BASE}/dumpstatus.json", status_path)
    with open(status_path) as f:
        status = json.load(f)
    files = status["jobs"]["articlesmultistreamdump"]["files"]
    # 只要正文分片，排除 index 文件
    shards = [
        (name, meta.get("size", 0))
        for name, meta in files.items()
        if "multistream" in name and "index" not in name and name.endswith(".bz2")
    ]
    shards.sort(key=lambda x: x[1])  # 小的先，省空间省时间
    return [name for name, _ in shards]


def extract_from_shard(shard: str, tmp_dir: Path) -> set[str]:
    dump = tmp_dir / shard
    print(f"[wiki] downloading {shard}", flush=True)
    _retrieve(f"{BASE}/{shard}", dump)
    urls: set[str] = set()
    try:
        with bz2.open(dump, "rt", errors="ignore") as f:
            for line in f:
                if refs := REF_RE.search(line):
                    urls.update(URL_RE.findall(refs.group(0)))
    finally:
        dump.unlink(missing_ok=True)  # 用完即删，控制峰值占用
    print(f"[wiki] {shard}: {len(urls)} unique urls", flush=True)
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("local-shared-data/enwiki_extracted_urls.txt"),
    )
    args = ap.parse_args()

    tmp_dir = Path("/tmp/wiki")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_urls: set[str] = set()
    shards = list_shards()[: args.n_shards]
    print(f"[wiki] selected {len(shards)} shards: {shards}", flush=True)
    for shard in shards:
        all_urls |= extract_from_shard(shard, tmp_dir)
        print(f"[wiki] running total: {len(all_urls)} unique urls", flush=True)

    with open(args.out, "w") as f:
        for url in sorted(all_urls):
            f.write(url + "\n")
    print(f"Done: wrote {len(all_urls)} urls to {args.out}", flush=True)


if __name__ == "__main__":
    main()