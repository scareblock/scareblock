# CLIP 모델 만들기 — 한 번만 하면 된다

04단계 시각 경로(피 · 거미 · 주사기)에 쓰는 사전학습 모델이다. **학습하지 않는다** — OpenAI가 공개한
CLIP ViT-B/32 가중치를 그대로 쓰고, 이미지 인코더만 ONNX로 바꾼다. `eval/clip_zeroshot.py`와
확장(`extension/fetch-deps.sh`)이 **같은 파일**을 읽는다.

**모델 파일은 저장소에 두지 않는다.** 이미지 인코더가 352 MB이고, 아래 절차로 공식 가중치에서 언제든
다시 만들 수 있다. 만든 결과는 `_local/models/clip/`에 둔다.

| 파일 | 크기 | 무엇 |
|---|---|---|
| `visual.onnx` | 351,783,456 B | 이미지 인코더 (fp32). 입력 `pixels` [n, 3, 224, 224] → 출력 `emb` [n, 512] |
| `text_emb.npy` | 8,320 B | 문장 4개의 임베딩(정규화) — 종류 3개 + `ordinary` |
| `meta.json` | 611 B | 문장 · 순서 · logit_scale · mean/std · 가중치 sha256 |

## 왜 ONNX로 바꾸나

공식 가중치를 읽으려면 PyTorch(1 GB 남짓)와 `clip` 패키지가 있어야 한다. `eval/`은 `onnxruntime` · `Pillow` ·
`numpy`로 돌고, 확장은 브라우저의 onnxruntime-web으로 돈다. **변환은 한 번, 실행은 계속**이므로
무거운 쪽을 한 번으로 몰아넣는다. YAMNet([yamnet-setup](yamnet-setup.md))과 같은 이유다.

**fp32 그대로 둔다.** int8 등으로 줄이면 점수가 달라져서, neutral-dev에서 정한 기준값 θ(#35 04단계)가 맞지 않는다.

## 1. 변환용 환경 (한 번)

파이썬 3.12로 만든다. 아래 버전으로 만들면 **2026.11.23에 쓴 파일과 바이트 단위로 같은 파일**이 나온다(4절).

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv _local/.venv-clip
_local/.venv-clip/bin/pip install torch==2.15.0 torchvision==0.29.2 onnx==1.23.2 onnxruntime==1.31.1 \
  numpy==2.5.4 pillow==13.0.0 ftfy regex tqdm packaging \
  "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
```

`clip`은 OpenAI 공식 저장소의 커밋을 고정해서 받는다. 다른 버전이어도 되지만, 그때는 4절의 sha256이 아니라
4절의 「값으로 확인」으로 본다.

**`Failed to build 'clip'`이 나면** (11.24, pip 26.2) — `clip`의 `setup.py`가 `pkg_resources`를 쓰는데, pip이 빌드할 때
새로 받는 setuptools(81 이상)에는 그게 없다. 나머지를 먼저 깔고, `clip`만 옛 setuptools로 따로 깐다.

```bash
_local/.venv-clip/bin/pip install torch==2.15.0 torchvision==0.29.2 onnx==1.23.2 onnxruntime==1.31.1 \
  numpy==2.5.4 pillow==13.0.0 ftfy regex tqdm packaging "setuptools<81" wheel
_local/.venv-clip/bin/pip install --no-build-isolation \
  "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
```

## 2. 변환 — 가중치 받기 · 문장 임베딩 · ONNX 내보내기

저장소 루트에서 돌린다. 처음 한 번은 `clip.load`가 공식 가중치를 받는다(353,976,522 B,
`openaipublic.azureedge.net/clip/models/40d3657…/ViT-B-32.pt`). 주소 안의 sha256과 받은 파일이
다르면 `clip.load`가 멈춘다.

```bash
mkdir -p _local/models/clip
_local/.venv-clip/bin/python - <<'PY'
"""CLIP ViT-B/32 (OpenAI 공식 가중치) → ONNX 이미지 인코더 + 고정 문장 임베딩. 한 번만 돌린다."""
import json, hashlib, numpy as np, torch, clip, onnxruntime as ort
from pathlib import Path
OUT = Path("_local/models/clip")
# 문장은 #35 11.05 등록 그대로 — 바꾸면 기준값 θ를 다시 정해야 한다
PROMPTS = {"blood": "a photo of blood", "syringe": "a photo of a syringe",
           "spider": "a photo of a spider", "ordinary": "an ordinary scene from a video"}
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False, download_root=str(OUT))
model = model.float().eval()
with torch.no_grad():
    tok = clip.tokenize(list(PROMPTS.values()))
    t = model.encode_text(tok); t = t / t.norm(dim=-1, keepdim=True)
np.save(OUT / "text_emb.npy", t.numpy().astype(np.float32))
x = torch.randn(2, 3, 224, 224)
torch.onnx.export(model.visual, x, OUT / "visual.onnx", input_names=["pixels"], output_names=["emb"],
                  dynamic_axes={"pixels": {0: "n"}, "emb": {0: "n"}}, opset_version=17, dynamo=False)
s = ort.InferenceSession(str(OUT / "visual.onnx"), providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = model.encode_image(x).numpy()
got = s.run(None, {"pixels": x.numpy()})[0]
pt = OUT / "ViT-B-32.pt"
meta = {"weights": pt.name, "weights_sha256": hashlib.sha256(pt.read_bytes()).hexdigest(),
        "prompts": PROMPTS, "order": list(PROMPTS), "logit_scale": float(model.logit_scale.exp()),
        "onnx_vs_torch_max_abs": float(np.abs(ref - got).max()),
        "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711],
        "size": 224, "resize": "shorter side 224 bicubic, center crop"}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print(json.dumps(meta, ensure_ascii=False, indent=1))
PY
```

몇 분 걸린다. 끝에 `meta.json` 내용이 찍힌다.

## 3. 확장에 넣기

```bash
bash extension/fetch-deps.sh     # models/ ← clip_visual.onnx · clip_heads.json
```

`clip_heads.json`은 `text_emb.npy` · `meta.json`과 **피 보강 모델 계수**(`probe.json`)를 묶은 것이다.

- **보강 모델은 저장소에 있다** — `extension/data/blood_probe.json`(약 12 KB). 스크립트가 기본으로 이것을 묶는다.
  가중치 512개 · 절편 · 학습 설정(C, 교차 검증 손실, 클립 번호)뿐이고 라벨이나 영상은 들어 있지 않다.
  리사이즈를 고친 뒤(#51) 다시 학습한 것이라 `config.CLIP_PROBE_THETA`(0.05868)와 짝이 맞다 (`C` = 10, `intercept` −6.674…)
- 다시 학습했으면(`eval/blood_probe train`, 라벨 · 영상 필요) 그 파일을 넘긴다

  ```bash
  PROBE=/다시/학습한/probe.json bash extension/fetch-deps.sh
  ```

- 파일이 없으면 피도 zero-shot으로 돈다. 스크립트가 「⚠ 보강 모델 없음」을 찍고 모든 종류를 `config.CLIP_THETA`로 판정한다

그다음 `chrome://extensions`에서 확장을 새로고침한다. 첫 로드는 모델이 커서 몇 초 걸린다.
실기 확인 절차는 `extension/README.md` 「CLIP 시각 경로」에 있다.

## 4. 같은 파일인지 확인한다

**바이트로 확인** — 1절 버전 그대로 만들었으면 `visual.onnx` · `text_emb.npy`는 같다 (2026.11.23, 11.24 혜인 노트북에서도 같음).
`meta.json`은 **다를 수 있다** — 2절의 검사 입력이 시드 없는 `torch.randn`이라 `onnx_vs_torch_max_abs`가 매번 조금 다르다
(11.23 1.2e-5 · 11.24 1.6e-5). 아래 「값으로 확인」으로 본다.

```bash
shasum -a 256 _local/models/clip/{visual.onnx,text_emb.npy,meta.json}
# 4a380c28356ae5223408842a05afab49f0cfcb48580d43b7e59213532bcf9cc9  visual.onnx
# 2a922507a401c5b2641a4c13a8b4f3f9e4a26c1ea6e8bd2bddc15f56cf8cc2c2  text_emb.npy
# eb8e6fe72bd329a8ddc019f81e50582ac036d932b7cc3ad0cacdbc3444adae06  meta.json  (11.23 — 위 이유로 다를 수 있다)
```

**값으로 확인** — 버전이 달라 바이트가 다르면 이것으로 본다.

- `meta.json`의 `weights_sha256`이 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`(공식 가중치)
- `onnx_vs_torch_max_abs`가 1e-4보다 작다 (11.23 파일은 1.2e-5) · `logit_scale`이 100
- 전처리가 eval과 바이트로 같은지:

  ```bash
  node extension/test/clip-prep.test.js      # 18/18 — 모델 · PIL이 없으면 「건너뜀」을 찍는다
  ```

- 브라우저 wasm으로 eval 점수와 같은지: `extension/test/clip-browser.html` (파일 안 주석대로 띄운다)

## 5. 변환 환경은 지워도 된다

`_local/.venv-clip`(약 940 MB)과 `_local/models/clip/ViT-B-32.pt`(338 MB)는 **다시 변환할 때만** 필요하다.
실행에는 `visual.onnx` · `text_emb.npy` · `meta.json` 셋이면 된다.
