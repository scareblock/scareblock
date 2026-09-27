# YAMNet 모델 만들기 — 한 번만 하면 된다

11.05 **조건부 열**(#35 결정 1)에 쓰는 사전학습 모델이다. **학습하지 않는다** — AudioSet으로
미리 학습된 클래스 점수를 그대로 읽는다.

**모델 파일은 저장소에 두지 않는다.** 15MB 이진 파일이고, 아래 절차로 공식 출처에서 언제든
다시 만들 수 있다. 만든 결과는 `_local/models/yamnet/yamnet.onnx`에 둔다.

## 왜 ONNX로 바꾸나

공식 가중치는 Keras(`.h5`)라 텐서플로가 있어야 열린다. 그런데 `eval/`의 의존성은 numpy·scipy
둘뿐이고, 여기에 텐서플로(1GB 남짓)를 얹고 싶지 않다. **변환은 한 번, 실행은 계속**이므로
무거운 쪽을 한 번으로 몰아넣는다. 변환 뒤에는 `onnxruntime`만 있으면 된다.

## 1. 변환용 환경 (한 번)

파이썬 3.12가 필요하다 — 텐서플로가 3.13 이후를 아직 지원하지 않는다.

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv _local/.venv-yamnet
_local/.venv-yamnet/bin/pip install tensorflow tf-keras tf2onnx onnx scipy
```

`tf-keras`가 필요한 이유 — 공식 YAMNet 코드가 `from tf_keras import ...`를 쓴다(Keras 2 API).

## 2. 공식 코드와 가중치

```bash
mkdir -p _local/models/yamnet && cd _local/models/yamnet
B=https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet
for f in params.py features.py yamnet.py yamnet_class_map.csv; do curl -sSfLO "$B/$f"; done
curl -sSfL -o yamnet.h5 https://storage.googleapis.com/audioset/yamnet.h5   # 15,296,092 B
```

둘 다 공식 출처다 — 코드는 `tensorflow/models`, 가중치는 구글 AudioSet 배포 주소.

## 3. SavedModel → ONNX

```bash
_local/.venv-yamnet/bin/python - <<'PY'
import sys, tensorflow as tf
sys.path.insert(0, "_local/models/yamnet")
import params as yp, yamnet as ym
m = ym.yamnet_frames_model(yp.Params())
m.load_weights("_local/models/yamnet/yamnet.h5")

@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")])
def infer(w):
    s, e, _ = m(w)
    return {"scores": s, "embeddings": e}

tf.saved_model.save(m, "_local/models/yamnet/saved", signatures={"serving_default": infer})
PY

_local/.venv-yamnet/bin/python -m tf2onnx.convert \
  --saved-model _local/models/yamnet/saved \
  --output _local/models/yamnet/yamnet.onnx --opset 17
```

## 4. 변환이 값을 바꾸지 않았는지 확인한다

같은 wav를 텐서플로와 ONNX에 넣어 상위 클래스를 비교한다. **2026.09.25 확인** —
`c001` 216.69초에서 `Cat 0.857 / 0.86`, `Caterwaul 0.745 / 0.75`, `Domestic animals 0.649 / 0.65`.

```bash
python3 -m eval.yamnet --selftest    # 모델 없이 도는 판정 로직 점검
python3 -m eval.yamnet --audio _local/dataset/wav --clips _local/dataset/clips.csv \
  --rate-sweep 0.05,0.1,0.2,0.4      # 임계값별 분당 발화율 (라벨 미사용)
```

## 5. 변환 환경은 지워도 된다

`_local/.venv-yamnet`(약 1.5GB)과 `_local/models/yamnet/saved`는 **다시 변환할 때만** 필요하다.
`yamnet.onnx`와 `yamnet_class_map.csv`만 남기면 실행에는 충분하다.

## 쓰는 클래스 — 라벨을 보기 전에 고정했다

`eval/yamnet.py`의 `CLASSES` 20개이고 근거는 [#35](https://github.com/scareblock/scareblock/issues/35)의
09.25 댓글에 있다. AudioSet 이름 뜻만 보고 골랐다 — 「갑자기 시작해서 놀라게 하는 소리」.
`Siren`은 우리 분류에서 지속 상태형이라 뺐다(제안서 부록 A.2).

**임계값도 라벨을 보기 전에 분당 발화율로 고른다** — 규칙 탐지기 동작점과 같은 규칙이다.
