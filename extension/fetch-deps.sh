#!/usr/bin/env bash
# 별도 문서 추론(offscreen.html)에 필요한 파일을 채운다. 둘 다 저장소에 두지 않는다.
#
#   vendor/  onnxruntime-web 1.20.1 (npm 공식 패키지) 중 wasm 실행에 쓰는 3개
#   models/  YAMNet — docs/yamnet-setup.md로 만든 _local/models/yamnet/ 에서 복사
#
# MV3는 원격 코드를 못 불러온다 — 런타임도 확장 안에 있어야 한다.
#   bash extension/fetch-deps.sh
set -euo pipefail

ORT_VERSION=1.20.1
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$HERE/vendor" "$HERE/models"

echo "onnxruntime-web@$ORT_VERSION 받는 중…"
(cd "$TMP" && npm pack "onnxruntime-web@$ORT_VERSION" --silent >/dev/null)
tar xzf "$TMP/onnxruntime-web-$ORT_VERSION.tgz" -C "$TMP"
for f in ort.wasm.min.js ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  cp "$TMP/package/dist/$f" "$HERE/vendor/$f"
done
echo "  vendor/ ← ort.wasm.min.js · ort-wasm-simd-threaded.{mjs,wasm}"

SRC="$ROOT/_local/models/yamnet"
if [[ -f "$SRC/yamnet.onnx" ]]; then
  cp "$SRC/yamnet.onnx" "$HERE/models/yamnet.onnx"
  [[ -f "$SRC/yamnet_class_map.csv" ]] && cp "$SRC/yamnet_class_map.csv" "$HERE/models/"
  echo "  models/ ← yamnet.onnx$( [[ -f "$SRC/yamnet_class_map.csv" ]] && echo ' · yamnet_class_map.csv')"
else
  echo "  ⚠ $SRC/yamnet.onnx 가 없다 — docs/yamnet-setup.md 대로 만든 뒤 이 스크립트를 다시 돌린다" >&2
  exit 1
fi
