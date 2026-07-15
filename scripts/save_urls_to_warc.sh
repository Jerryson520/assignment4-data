wget \
  --timeout=5 \
  --read-timeout=10 \
  --tries=1 \
  --max-redirect=5 \
  --warc-file=quality_data/wiki_positive \
  --input-file=quality_data/wiki_urls.txt \
  --output-document=/dev/null \
  --no-verbose