#!/usr/bin/env bash
# Hailo Model Zoo 가 HEF 를 컴파일한 원본 float ONNX 를 models/onnx/ 에 받는다 — Pi 의 HEF 와 같은 가중치.
# 첫 실행(SHA256SUMS 없음): zip 크기·멤버 크기/CRC32 를 확인하고 해시를 출력한 뒤 3 으로 끝난다 → models/onnx/SHA256SUMS 로 저장해 커밋.
# 그 뒤로는 SHA256SUMS 검증만 한다. zip 은 받아 두고(git 제외) ONNX 만 꺼낸다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/../models/onnx"
BASE="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com"
# zip 경로 | zip 크기 | zip 안의 onnx | onnx 크기 | onnx CRC32 (2026-10-28 S3 HEAD·중앙 디렉터리로 확인)
SOURCES=(
  "FaceDetection/scrfd/scrfd_10g/pretrained/2022-09-07/scrfd_10g.zip|15714580|scrfd_10g_bnkps.onnx|16923479|552e1063"
  "FaceRecognition/arcface/arcface_mobilefacenet/pretrained/2022-08-24/arcface_mobilefacenet.zip|7696419|mbf.onnx|8179605|1e758c1f"
)
mkdir -p "$DEST"
if [[ -f "$DEST/SHA256SUMS" ]] && (cd "$DEST" && sha256sum -c --quiet SHA256SUMS 2>/dev/null); then
    echo "ok      models/onnx"; exit 0
fi
for entry in "${SOURCES[@]}"; do
    IFS='|' read -r url_path zip_size onnx onnx_size onnx_crc <<<"$entry"
    zip="$DEST/$(basename "$url_path")"
    if [[ ! -f "$zip" || "$(stat -c %s "$zip")" != "$zip_size" ]]; then
        echo "download $(basename "$url_path")"
        curl --fail --location --continue-at - --output "$zip.tmp" "$BASE/$url_path"
        mv "$zip.tmp" "$zip"
    fi
    [[ "$(stat -c %s "$zip")" == "$zip_size" ]] || { echo "$zip: 크기 $(stat -c %s "$zip") != $zip_size" >&2; exit 2; }
    # 멤버 크기·CRC32 는 중앙 디렉터리 값으로 확인한다 (unzip -lv: Length Method Size Cmpr Date Time CRC-32 Name)
    unzip -lv "$zip" | awk -v n="$onnx" -v sz="$onnx_size" -v crc="$onnx_crc" \
        '$NF ~ n"$" { if ($1 == sz && tolower($7) == crc) ok = 1 } END { exit ok ? 0 : 1 }' \
        || { echo "$zip: $onnx 의 크기/CRC 가 기대값($onnx_size/$onnx_crc)과 다릅니다" >&2; unzip -lv "$zip" >&2; exit 2; }
    unzip -o -j -q "$zip" "*$onnx" -d "$DEST"
    [[ -f "$DEST/$onnx" ]] || { echo "$onnx 가 zip 에 없습니다" >&2; exit 2; }
done
if [[ -f "$DEST/SHA256SUMS" ]]; then
    (cd "$DEST" && sha256sum -c SHA256SUMS)
else
    echo "SHA256SUMS 가 없습니다 — 아래를 models/onnx/SHA256SUMS 로 저장하고 커밋하세요:" >&2
    (cd "$DEST" && sha256sum *.zip *.onnx)
    exit 3
fi
