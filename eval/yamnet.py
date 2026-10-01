"""YAMNet 사전학습 클래스 점수를 탐지 경로 한 열로 — **학습하지 않는다**.

11.05 조건부 열(#35 결정 1). 규칙 탐지기와 **같은 지연 정의**로 재서 E1 표에 나란히 놓는다.

사용:
    python3 -m eval.yamnet --audio _local/dataset/wav --clips _local/dataset/clips.csv \\
        --rate-sweep 0.05,0.1,0.2,0.4        # 임계값별 분당 발화율 — 라벨을 읽지 않는다
    python3 -m eval.yamnet --selftest        # 모델 없이 도는 판정 로직 점검

**모델은 저장소에 없다.** `docs/yamnet-setup.md`의 절차로 `_local/models/yamnet/yamnet.onnx`를
만든다(공식 가중치 → ONNX). 변환은 한 번이고, 그 뒤로는 onnxruntime만 있으면 된다.

핵심 제약 — YAMNet은 **패치 0.96초(실제로 덮는 오디오 0.975초)를 0.48초씩** 민다.
규칙 탐지기(0.025초 창·0.01초 간격)보다 구조적으로 **0.975초** 느리고, 임계값을 어떻게
잡아도 줄지 않는다. 이 열의 값은 「누가 더 정확한가」가
아니라 **「이름을 붙일 수 있지만 느리다 vs 이름은 못 붙여도 빠르다」**를 같은 지연 축에서
보이는 데 있다.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import labels as L
from .detect import Detection, detections_per_minute
from .features import load_wav

MODEL = Path("_local/models/yamnet/yamnet.onnx")
PATCH_S = 0.96     # YAMNet 패치 창 (공식 params.patch_window_seconds)
STFT_WIN_S = 0.025 # params.stft_window_seconds
STFT_HOP_S = 0.010 # params.stft_hop_seconds
# 패치는 STFT 프레임 96개고, **실제로 덮는 오디오는 0.975초**다 —
# (96−1)×0.010 + 0.025. 공식 features.pad_waveform도 최소 길이를 이 식으로 잡는다.
# 확신 시각을 0.96으로 잡으면 모델이 본 마지막 샘플보다 15 ms 앞서고, 그러면
# 「규칙 탐지기의 t_ready와 같은 정의」라는 이 열의 전제가 깨진다 (#43 리뷰 🟡1).
WIN_S = round((PATCH_S / STFT_HOP_S - 1) * STFT_HOP_S + STFT_WIN_S, 6)   # 0.975
HOP_S = 0.48      # 창 간격
SR = 16000        # 모델이 기대하는 표본율. clipcut이 뽑는 wav와 같다

# #35에 사전 등록(09.25) — AudioSet 이름 뜻만 보고 고른 「갑자기 시작해서 놀라게 하는 소리」.
# **라벨을 보고 고르지 않았다.** Siren은 우리 분류에서 지속 상태형이라 뺐다(부록 A.2).
CLASSES: dict[int, str] = {
    6: "Shout", 9: "Yell", 11: "Screaming", 74: "Growling", 80: "Caterwaul",
    105: "Roar", 281: "Thunder", 348: "Door", 352: "Slam", 353: "Knock",
    420: "Explosion", 421: "Gunshot, gunfire", 430: "Boom", 435: "Glass",
    437: "Shatter", 454: "Thump, thud", 460: "Bang", 462: "Whack, thwack",
    463: "Smash, crash", 464: "Breaking",
}


@dataclass
class Session:
    """onnxruntime 세션. 임포트를 지연시켜 `eval`의 나머지가 onnxruntime 없이도 돈다."""
    sess: object
    out_names: list[str]

    @classmethod
    def load(cls, model: Path) -> "Session":
        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover - 환경 문제
            raise SystemExit("onnxruntime이 없다 — pip install onnxruntime "
                             "(eval의 나머지는 이것 없이 돈다)")
        if not model.exists():
            raise SystemExit(f"모델이 없다: {model}\n"
                             f"docs/yamnet-setup.md 절차로 만든다 (변환은 한 번뿐이다)")
        s = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        return cls(s, [o.name for o in s.get_outputs()])

    def scores(self, wav: Path) -> np.ndarray:
        """클립 하나의 프레임별 점수 (프레임 × 521)."""
        x, sr = load_wav(wav)
        if sr != SR:
            raise SystemExit(f"{wav}: {sr}Hz — YAMNet은 {SR}Hz를 기대한다")
        out = self.sess.run(None, {"waveform": np.asarray(x, dtype=np.float32)})
        return out[self.out_names.index("scores")]


def subset_max(scores: np.ndarray) -> np.ndarray:
    """프레임마다 **사전 등록한 클래스들 중 최댓값**. 이 값 하나로 판정한다."""
    idx = sorted(CLASSES)
    return scores[:, idx].max(axis=1)


def to_detections(peak: np.ndarray, threshold: float, clip_id: str = "") -> list[Detection]:
    """임계값을 넘는 프레임을 탐지로 바꾼다.

    **연속 프레임은 한 건으로 묶는다.** 0.48초 간격이라 한 사건이 두세 프레임에 걸치는데,
    그걸 따로 세면 발화율이 사건 수가 아니라 사건 길이를 재게 된다.

    시각의 정의를 규칙 탐지기와 맞춘다.
      - `fire`(확신) = **그 창의 끝** — `detect.run`이 `t_ready`(분석 창의 끝)를 쓰는 것과 같다
      - `onset`(시작 추정) = **묶음 첫 창의 시작** — YAMNet은 되짚기를 하지 않으므로
        창 경계가 말할 수 있는 전부다. 해상도가 0.48초라 ±0.5초 매칭(§6)과 같은 눈금이다

    `source="yamnet"`으로 표시한다. **`onset_capped`는 쓰지 않는다** — 그 필드는 「되짚기가
    상한에 걸렸다 = 추정 실패」라는 뜻이고(`detect.py`), 리포트가 그걸 「onset 되짚기 상한 N건」
    으로 센다. YAMNet은 되짚기를 아예 하지 않으므로 그 칸에 들어가면 **되짚기 실패율로 읽힌다**
    (#43 리뷰 🟡2).
    """
    out: list[Detection] = []
    i, n = 0, len(peak)
    while i < n:
        if peak[i] < threshold:
            i += 1
            continue
        j = i
        while j + 1 < n and peak[j + 1] >= threshold:
            j += 1
        out.append(Detection(onset=i * HOP_S, fire=j * HOP_S + WIN_S,
                             score=float(peak[i:j + 1].max()),
                             clip_id=clip_id, source="yamnet"))
        i = j + 1
    return out


def sweep(clip_peaks: dict[str, np.ndarray], thresholds: list[float], total_s: float):
    """임계값별 (탐지 수, 건/분). **라벨을 읽지 않는다** — 동작점과 같은 규칙이다."""
    rows = []
    for t in thresholds:
        dets = [d for cid, pk in clip_peaks.items() for d in to_detections(pk, t, cid)]
        rows.append((t, len(dets), detections_per_minute(dets, total_s)))
    return rows


def _selftest() -> int:
    """모델 없이 판정 로직만 본다 — 이 부분이 틀리면 지연이 통째로 어긋난다."""
    ok = True

    # ① 연속 프레임은 한 건. 시각은 창 경계로.
    pk = np.array([0.0, 0.7, 0.8, 0.0, 0.0, 0.9])
    d = to_detections(pk, 0.5, "c1")
    want_n, want = 2, [(1 * HOP_S, 2 * HOP_S + WIN_S), (5 * HOP_S, 5 * HOP_S + WIN_S)]
    got = [(x.onset, x.fire) for x in d]
    merge_ok = len(d) == want_n and all(
        abs(a - b) < 1e-9 and abs(c - e) < 1e-9
        for (a, c), (b, e) in zip(got, want))
    print(f"묶기: {len(d)}건 · {[(round(a,2), round(b,2)) for a, b in got]} "
          f"{'통과' if merge_ok else '실패'}")
    ok &= merge_ok

    # ② 확신 시각은 창의 끝이다 — 시작이 아니다. 여기가 어긋나면 지연이 0.975초 빨라 보인다
    lag_ok = abs(d[1].fire - d[1].onset - WIN_S) < 1e-9 and d[1].fire > d[1].onset
    print(f"확신 시각: onset {d[1].onset:.2f} → fire {d[1].fire:.2f} "
          f"(창 {WIN_S}초 뒤) {'통과' if lag_ok else '실패'}")
    ok &= lag_ok

    # ③ 임계값이 올라가면 탐지가 줄어든다(단조).
    # 봉우리를 떨어뜨려 놓는다 — 붙여 놓으면 묶기가 전부 1건으로 만들어 이 검사가 아무것도
    # 지키지 않는다(처음 픽스처가 그랬다: [1,1,1,1,0]).
    pk2 = np.array([0.9, 0.0, 0.6, 0.0, 0.35, 0.0, 0.1])
    counts = [len(to_detections(pk2, t)) for t in (0.05, 0.3, 0.5, 0.8, 0.95)]
    mono_ok = counts == [4, 3, 2, 1, 0]
    print(f"임계값 단조: {counts} {'통과' if mono_ok else '실패'}")
    ok &= mono_ok

    # ④ 사전 등록한 클래스만 본다. 목록 밖이 아무리 높아도 반응하지 않는다
    s = np.zeros((2, 521), dtype=np.float32)
    s[0, 11] = 0.7          # Screaming — 목록 안
    s[1, 0] = 0.99          # Speech — 목록 밖
    peak = subset_max(s)
    subset_ok = abs(peak[0] - 0.7) < 1e-6 and peak[1] < 1e-6
    print(f"클래스 한정: 목록 안 {peak[0]:.2f} · 목록 밖 {peak[1]:.2f} "
          f"{'통과' if subset_ok else '실패'}")
    ok &= subset_ok

    # ⑤ 덮는 길이는 패치 0.96초가 아니라 0.975초다 — 여기가 되돌아가면 YAMNet 지연이
    #    전부 15 ms 짧게 나오고 「규칙 탐지기와 같은 정의」가 아니게 된다 (#43 리뷰 🟡1)
    cover_ok = abs(WIN_S - 0.975) < 1e-9 and WIN_S > PATCH_S
    print(f"덮는 오디오: 패치 {PATCH_S}초 → {WIN_S}초 {'통과' if cover_ok else '실패'}")
    ok &= cover_ok

    # ⑥ source로 표시하고 onset_capped는 건드리지 않는다 — 그 필드는 「되짚기 실패」다 (🟡2)
    src_ok = all(x.source == "yamnet" and not x.onset_capped for x in d)
    print(f"출처 표시: source={d[0].source} · onset_capped={d[0].onset_capped} "
          f"{'통과' if src_ok else '실패'}")
    ok &= src_ok

    # ⑦ 사전 등록 목록이 바뀌지 않았다 (#35 09.25)
    reg_ok = len(CLASSES) == 20 and 11 in CLASSES and 420 in CLASSES and 0 not in CLASSES
    print(f"등록 목록: {len(CLASSES)}개 {'통과' if reg_ok else '실패'}")
    ok &= reg_ok

    print("\n자체 점검:", "통과" if ok else "실패")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="YAMNet 사전학습 클래스 점수 — 조건부 열")
    ap.add_argument("--audio", help="clip_id.wav 들이 있는 디렉터리")
    ap.add_argument("--clips", help="클립 명세 CSV — 없으면 wav 디렉터리 전체")
    ap.add_argument("--model", default=str(MODEL), help=f"ONNX 모델 (기본 {MODEL})")
    ap.add_argument("--rate-sweep", metavar="값,값,…",
                    help="임계값별 분당 발화율 표. 라벨을 읽지 않는다")
    ap.add_argument("--labels", help=argparse.SUPPRESS)   # 거부용으로만 받는다
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if a.labels:
        ap.error("이 도구는 라벨을 읽지 않는다 — 임계값은 라벨을 보기 전에 발화율로 고른다")
    if not a.audio or not a.rate_sweep:
        ap.error("--audio 와 --rate-sweep 이 필요하다 (또는 --selftest)")
    try:
        ths = [float(v) for v in a.rate_sweep.split(",")]
    except ValueError:
        ap.error(f"--rate-sweep 값은 숫자여야 한다 — 받은 것: {a.rate_sweep}")
    if any(t <= 0 or t > 1 for t in ths):
        ap.error("임계값은 0 초과 1 이하다 — YAMNet 점수는 0~1이다")

    audio = {p.stem: p for p in Path(a.audio).glob("*.wav")}
    ids = L.load_clips(a.clips) if a.clips else sorted(audio)
    used = {c: audio[c] for c in ids if c in audio}
    if not used:
        print(f"실패: 평가할 wav가 없다 — --audio {a.audio}", file=sys.stderr)
        return 1
    missing = [c for c in ids if c not in audio]
    if missing:
        print(f"경고: wav 없는 클립 {missing}", file=sys.stderr)

    sess = Session.load(Path(a.model))
    peaks, total_s = {}, 0.0
    for k, (cid, wav) in enumerate(sorted(used.items()), start=1):
        x, sr = load_wav(wav)
        total_s += len(x) / sr
        peaks[cid] = subset_max(sess.scores(wav))
        print(f"\r  점수 계산 {k}/{len(used)} …", end="", file=sys.stderr, flush=True)
    print("\r" + " " * 30 + "\r", end="", file=sys.stderr)

    rows = sweep(peaks, ths, total_s)
    print(f"클립 {len(used)}/{len(ids)}개 · {total_s / 60:.1f}분 · YAMNet 사전학습 "
          f"{len(CLASSES)}개 클래스 최댓값 · 라벨 미사용 — 오탐률이 아니라 발화율")
    print(f"패치 {PATCH_S}초(덮는 오디오 {WIN_S}초) · 간격 {HOP_S}초 — 확신 시각은 창의 끝이라 "
          f"**구조적으로 {WIN_S}초 늦는다**\n")
    print("| 임계값 | 탐지 | 건/분 |\n|---|---|---|")
    for t, n, r in rows:
        print(f"| {t:g} | {n} | {r:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
