"""#40 새 기준 — 「말소리면 버린다」. #35 10.03 사전 등록 그대로.

규칙 탐지(동작점 20)가 울리면 사건 시작부터 YAMNet 패치 하나(0.975초)를 보고,
`Speech ≥ θ` **이고** `Speech > #43 등록 20개 클래스의 최댓값`이면 기각한다.
확신 시각은 패치 끝(onset + 0.975초)으로 민다. **학습하지 않는다.**

사용:
    # 개발 세트 — θ 후보 0.1…0.9를 재고 등록한 규칙으로 하나를 고른다
    python3 -m eval.speech_gate dev --labels … --audio … --neutral-audio … --split-dir …
    # 시험 세트 — 위에서 고른 θ로 **한 번만**. θ 찾기는 하지 않는다
    python3 -m eval.speech_gate test --theta 0.5 …
    python3 -m eval.speech_gate --selftest

**시험 세트에는 θ 스윕이 없다.** 시험 세트를 보고 θ를 고르면 사전 등록의 의미가 없어진다.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import labels as L
from .detect import Detection, Params, detections_per_minute, run
from .evaluate import RENDER_S, latency_summary, score
from .features import extract, load_wav

SPEECH = 0                      # AudioSet 클래스 0 = Speech
THETAS = [round(0.1 * k, 1) for k in range(1, 10)]
KEEP_RECALL = 0.90              # 개발 세트: 참 사건 매칭 수가 「검증 없음」의 90% 이상
SURGE = 20.0                    # #35 09.22 사전 등록 동작점
LOOKAHEAD = 3.0


def gate(dets: list[Detection], x: np.ndarray, sr: int, yam, theta: float | None,
         win_s: float, scare_idx: list[int]) -> tuple[list[Detection], int, int]:
    """(남은 탐지, 기각 수, 관찰 못 함 수). theta가 None이면 기각하지 않고 확신 시각만 민다."""
    out, rejected, unseen = [], 0, 0
    for d in dets:
        a, b = int(round(d.onset * sr)), int(round((d.onset + win_s) * sr))
        if b > len(x):
            # 클립이 먼저 끝나 패치를 다 못 본다 — 기각하지 않는다(#40 클립 끝 처리와 같은 이유)
            unseen += 1
            out.append(replace(d, verify_capped=True))
            continue
        s = yam(x[a:b])
        speech, scare = float(s[SPEECH]), float(s[scare_idx].max())
        if theta is not None and speech >= theta and speech > scare:
            rejected += 1
            continue
        out.append(replace(d, fire=max(d.fire, d.onset + win_s)))
    return out, rejected, unseen


def _yamnet(model: Path):
    from .yamnet import Session, SR, WIN_S, CLASSES
    sess = Session.load(model)

    def one(snippet: np.ndarray) -> np.ndarray:
        out = sess.sess.run(None, {"waveform": np.asarray(snippet, dtype=np.float32)})
        sc = out[sess.out_names.index("scores")]
        return sc[0]                      # 0.975초 = 패치 하나
    return one, SR, WIN_S, sorted(CLASSES)


def _clip_list(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


def measure(clips: list[str], audio: Path, truth: list[L.Label] | None, yam, sr_exp, win_s,
            scare_idx, thetas: list[float | None]):
    """θ마다 (Metrics 또는 None, 건/분, 기각, 관찰 못 함). truth가 None이면 발화율만."""
    base = Params(surge_of_quiet=SURGE)
    per_clip = []
    total_s = 0.0
    for cid in clips:
        x, sr = load_wav(audio / f"{cid}.wav")
        if sr != sr_exp:
            raise SystemExit(f"{cid}: {sr}Hz — YAMNet은 {sr_exp}Hz")
        dets = run(extract(x, sr), base, clip_id=cid)
        per_clip.append((cid, x, sr, dets))
        total_s += len(x) / sr
    # 패치 점수는 θ와 무관하다 — 한 번만 계산해 둔다
    cache: dict[tuple[str, float], np.ndarray] = {}

    def cached(cid):
        def f(snip):
            k = (cid, hash(np.asarray(snip).tobytes()))
            if k not in cache:
                cache[k] = yam(snip)
            return cache[k]
        return f

    res = {}
    for th in thetas:
        kept, rej, uns = [], 0, 0
        for cid, x, sr, dets in per_clip:
            k, r, u = gate(dets, x, sr, cached(cid), th, win_s, scare_idx)
            kept += k; rej += r; uns += u
        m = None
        if truth is not None:
            t = [y for y in truth if y.clip_id in set(clips) and y.category == "jumpscare"]
            m = score(t, kept, LOOKAHEAD, render_s=RENDER_S)
        res[th] = (m, detections_per_minute(kept, total_s), rej, uns)
    return res, total_s


def pick(dev_h: dict, dev_n: dict, base_tp: int) -> float | None:
    """등록한 규칙: 매칭 수가 「검증 없음」의 90% 이상인 θ 중 일반 발화율 최소, 동점이면 작은 θ."""
    ok = [th for th in THETAS if dev_h[th][0].tp >= KEEP_RECALL * base_tp]
    if not ok:
        return None
    return min(ok, key=lambda th: (round(dev_n[th][1], 6), th))


def _row(name, m, rate_h, rate_n, rej_h, rej_n):
    p = m.tp / m.n_pred if m.n_pred else 0.0
    r = m.tp / m.n_true if m.n_true else 0.0
    lat = latency_summary(m.latencies) if m.latencies else None
    p90 = f"{lat['p90']:.3f}" if lat else "—"
    return (f"| {name} | {m.tp} | {r:.3f} | {p:.3f} | {p90} | {rate_h:.2f} | {rate_n:.2f} "
            f"| {rej_h} / {rej_n} |")


HEAD = ("| θ | 매칭 | recall | precision | 지연 p90 (s) | 공포 건/분 | 일반 건/분 | 기각 공포/일반 |\n"
        "|---|---|---|---|---|---|---|---|")


def _selftest() -> int:
    # 모델 없이 판정 로직만: 가짜 점수 함수
    sr, win = 16000, 0.975
    x = np.zeros(sr * 10, dtype=np.float32)
    mk = lambda o: Detection(onset=o, fire=o + 0.1, score=1.0, clip_id="t")
    dets = [mk(1.0), mk(3.0), mk(5.0), mk(9.5)]
    table = {16000: "speech", 48000: "scream", 80000: "quiet"}
    # 위치로 종류를 고르는 가짜 — 패치 시작 표본으로 판정
    def fake_by_pos(pos):
        def f(snip):
            s = np.zeros(521, dtype=np.float32)
            k = table.get(pos[0])
            if k == "speech": s[SPEECH] = 0.8; s[11] = 0.1
            if k == "scream": s[SPEECH] = 0.6; s[11] = 0.9
            if k == "quiet":  s[SPEECH] = 0.05
            return s
        return f
    kept = []
    rej = unseen = 0
    for d in dets:
        pos = [int(round(d.onset * sr))]
        k, r, u = gate([d], x, sr, fake_by_pos(pos), 0.5, win, [11])
        kept += k; rej += r; unseen += u
    ok_rej = rej == 1                                   # 말소리만 기각
    ok_scream = any(abs(d.onset - 3.0) < 1e-9 for d in kept)   # 비명은 Speech가 커도 남김
    ok_unseen = unseen == 1 and any(d.verify_capped for d in kept)   # 9.5초 — 끝을 넘는다
    ok_fire = all(d.fire >= d.onset + win - 1e-9 for d in kept if not d.verify_capped)
    k0, r0, _ = gate(dets, x, sr, fake_by_pos([16000]), None, win, [11])
    ok_none = r0 == 0 and len(k0) == len(dets)
    ok = ok_rej and ok_scream and ok_unseen and ok_fire and ok_none
    print(f"말소리 기각 {'통과' if ok_rej else '실패'} · 비명 보존 {'통과' if ok_scream else '실패'} · "
          f"클립 끝 관찰 못 함 {'통과' if ok_unseen else '실패'} · 확신 시각 밀기 {'통과' if ok_fire else '실패'} · "
          f"θ 없음=기각 없음 {'통과' if ok_none else '실패'}")
    print("자체 점검:", "통과" if ok else "실패")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="#40 말소리 기각 기준 (#35 10.03 사전 등록)")
    ap.add_argument("mode", nargs="?", choices=["dev", "test"])
    ap.add_argument("--labels"); ap.add_argument("--audio"); ap.add_argument("--neutral-audio")
    ap.add_argument("--split-dir", help="horror-dev.txt · horror-test.txt · neutral-dev.txt · neutral-test.txt")
    ap.add_argument("--model", default="_local/models/yamnet/yamnet.onnx")
    ap.add_argument("--theta", type=float, help="test 모드 전용 — 개발 세트에서 고른 값")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.mode:
        ap.error("dev 또는 test")
    if a.mode == "test" and a.theta is None:
        ap.error("test 모드는 --theta가 필요하다 — 개발 세트에서 고른 값을 넣는다")
    if a.mode == "dev" and a.theta is not None:
        ap.error("dev 모드는 θ를 받지 않는다 — 후보 전체를 재서 등록한 규칙으로 고른다")

    sd = Path(a.split_dir)
    hc = _clip_list(sd / f"horror-{a.mode}.txt")
    nc = _clip_list(sd / f"neutral-{a.mode}.txt")
    truth = L.pick_annotator(L.load(a.labels), "B")
    yam, sr_exp, win_s, scare_idx = _yamnet(Path(a.model))

    thetas = [None] + (THETAS if a.mode == "dev" else [a.theta])
    # None = 기각 없이 패치만 기다림(지연 비교용). 검증 자체를 끈 기준선은 아래 off
    h, h_s = measure(hc, Path(a.audio), truth, yam, sr_exp, win_s, scare_idx, thetas)
    n, n_s = measure(nc, Path(a.neutral_audio), None, yam, sr_exp, win_s, scare_idx, thetas)

    # 검증 없음(지금의 E1 탐지기) — 같은 클립에서
    off_h = measure(hc, Path(a.audio), truth, lambda s: np.zeros(521), sr_exp, 0.0, scare_idx, [None])[0][None]
    off_n = measure(nc, Path(a.neutral_audio), None, lambda s: np.zeros(521), sr_exp, 0.0, scare_idx, [None])[0][None]

    print(f"{a.mode} 세트 · 공포 {len(hc)}클립 {h_s/60:.1f}분 · 일반 {len(nc)}클립 {n_s/60:.1f}분 · "
          f"동작점 {SURGE:g} · 렌더링 {RENDER_S}s · 관찰 못 함 공포 {h[None][3]} · 일반 {n[None][3]}")
    print(HEAD)
    print(_row("검증 없음", off_h[0], off_h[1], off_n[1], 0, 0))
    print(_row("패치만 기다림", h[None][0], h[None][1], n[None][1], 0, 0))
    for th in thetas[1:]:
        print(_row(f"{th:.1f}", h[th][0], h[th][1], n[th][1], h[th][2], n[th][2]))
    if a.mode == "dev":
        th = pick(h, n, off_h[0].tp)
        if th is None:
            print("\n→ 90% 조건을 만족하는 θ가 없다 — 등록한 규칙상 이 기준은 채택하지 않는다")
        else:
            print(f"\n→ 등록한 규칙으로 고른 θ = {th:.1f} — #35에 적은 뒤 test 모드로 한 번만 돌린다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
