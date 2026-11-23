#!/usr/bin/env bash
# 별도 문서 추론(offscreen.html)에 필요한 파일을 채운다. 둘 다 저장소에 두지 않는다.
#
#   vendor/  onnxruntime-web 1.20.1 (npm 공식 패키지) 중 wasm 실행에 쓰는 3개
#   models/  YAMNet — docs/yamnet-setup.md로 만든 _local/models/yamnet/ 에서 복사
#            CLIP  — docs/clip-setup.md로 만든 _local/models/clip/ (eval/clip_zeroshot과 같은 파일) + 피 보강 모델 계수
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

# CLIP ViT-B/32 이미지 쪽 — eval/clip_zeroshot이 쓰는 같은 파일(fp32, 352 MB). 없으면 CLIP만 꺼지고 나머지는 돈다.
# heads(문장 임베딩 · logit_scale · 피 보강 모델 계수)는 numpy로 한 JSON으로 묶는다.
CLIP_SRC="$ROOT/_local/models/clip"
# eval/blood_probe train 결과 (#53). 리사이즈를 버림으로 고친 뒤 다시 학습한 것 — config.CLIP_PROBE_THETA와 짝이다
PROBE="${PROBE:-$ROOT/_local/dataset/resize-1123/probe.json}"
if [[ -f "$CLIP_SRC/visual.onnx" && -f "$CLIP_SRC/text_emb.npy" ]]; then
  cp "$CLIP_SRC/visual.onnx" "$HERE/models/clip_visual.onnx"
  python3 - "$CLIP_SRC" "$PROBE" "$HERE/models/clip_heads.json" <<'PY'
import json, sys, numpy as np
src, probe, out = sys.argv[1:4]
meta = json.load(open(f"{src}/meta.json"))
heads = dict(order=meta["order"], logit_scale=meta["logit_scale"], prompts=meta["prompts"],
             weights_sha256=meta["weights_sha256"],
             text_emb=np.load(f"{src}/text_emb.npy").astype(np.float32).tolist())
try:
    p = json.load(open(probe))
    heads["probe"] = dict(cat="blood", coef=p["coef"], intercept=p["intercept"], C=p["C"])
except FileNotFoundError:
    print("  ⚠ 보강 모델 없음 — 피도 zero-shot으로 돈다", file=sys.stderr)
json.dump(heads, open(out, "w"))
PY
  echo "  models/ ← clip_visual.onnx · clip_heads.json$( [[ -f "$PROBE" ]] && echo ' (피 보강 모델 포함)')"
else
  echo "  ⚠ $CLIP_SRC/visual.onnx 가 없다 — CLIP 없이 돈다 (만들려면 docs/clip-setup.md)" >&2
fi
