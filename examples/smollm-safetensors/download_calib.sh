#!/usr/bin/env bash
# Calibration text for SmolLM2's static W8A8 activation scales: the first DOCS
# documents of one pinned C4 shard (allenai/c4, English validation split,
# ODC-BY), written to data/calib/smollm.txt. Never committed (data/calib is
# ignored). Run from anywhere:
#   bash examples/smollm-safetensors/download_calib.sh [DOCS]     # DOCS defaults to 200
set -euo pipefail

REVISION=1588ec454efa1a09f29cd18ddd04fe05fc8653a2
SHARD=en/c4-validation.00000-of-00008.json.gz
BYTES=40471190
SHA256=1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f
URL="https://huggingface.co/datasets/allenai/c4/resolve/${REVISION}/${SHARD}"
DOCS="${1:-200}"

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${repo}/data/calib/smollm.txt"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "downloading ${SHARD} (${BYTES} bytes) at ${REVISION:0:7}"
curl --fail --location --retry 3 --silent --show-error --output "$tmp/shard.json.gz" "$URL"

# Fail closed: a shard of another size or hash is never used.
[ "$(wc -c < "$tmp/shard.json.gz" | tr -d ' ')" = "$BYTES" ] || { echo "size mismatch" >&2; exit 1; }
echo "${SHA256}  $tmp/shard.json.gz" | sha256sum --check --quiet

# One JSON object per line; keep the "text" of the first DOCS, a blank line apart.
python3 - "$tmp/shard.json.gz" "$DOCS" "$tmp/smollm.txt" <<'PY'
import gzip, json, sys
src, docs, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
texts = []
with gzip.open(src, 'rt', encoding='utf-8') as f:
    for line in f:
        texts.append(json.loads(line)['text'].strip())
        if len(texts) == docs:
            break
with open(dst, 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n\n'.join(texts) + '\n')
print(f'{len(texts)} documents, {sum(map(len, texts))} characters')
PY

mkdir -p "$(dirname "$out")"
mv "$tmp/smollm.txt" "$out"
echo "wrote ${out#"$repo"/}"
