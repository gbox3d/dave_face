#!/usr/bin/env bash
# Model Zoo v5.4.0 hailo10h HEF 를 models/5.4.0/ 에 받고 SHA-256 을 검증한다.
# HEF_CACHE 디렉터리에 해시가 맞는 사본이 있으면 다운로드 대신 복사한다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../models/5.4.0"
BASE_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h"
CACHE="${HEF_CACHE:-}"

verified() { echo "$1  $2" | sha256sum -c --quiet - 2>/dev/null; }

while read -r sum name; do
    [[ -z "$sum" ]] && continue
    target="$DEST/$name"
    if [[ -f "$target" ]] && verified "$sum" "$target"; then
        echo "ok      $name"
    elif [[ -n "$CACHE" && -f "$CACHE/$name" ]] && verified "$sum" "$CACHE/$name"; then
        cp "$CACHE/$name" "$target"
        echo "copied  $name"
    else
        echo "download $name"
        curl --fail --location --continue-at - --output "$target.tmp" "$BASE_URL/$name"
        mv "$target.tmp" "$target"
    fi
done < "$DEST/SHA256SUMS"

(cd "$DEST" && sha256sum -c SHA256SUMS)
