"""사이렌 — YAMNet 기존 클래스에 임계값만 (#35 11.23 등록). **학습하지 않는다.**

    python3 -m eval.siren score --wav DIR [DIR …] --out DIR          # 클립마다 패치 점수 .npy
    python3 -m eval.siren evaluate --scores DIR --labels labels-S.csv --clips clips.csv \\
        --neutral-dev dev.txt --neutral-test test.txt --out result.json
    python3 -m eval.siren --selftest

**정의 (결과를 보기 전에 등록한 그대로)**

- 패치: `eval/yamnet.py`와 같다 — 0.975 s 창, 0.48 s 간격, 패치 k의 창은 `[0.48·k, 0.48·k + 0.975)`
- 점수 `p_k` = 클래스 390 Siren · 317 Police car · 318 Ambulance · 319 Fire engine 점수의 최댓값
  (391 Civil defense siren은 넣지 않는다 — 공습경보는 가이드에서 미결)
- 가림: `p_k ≥ θ`인 패치는 그 창을 가린다. 확신 시각 = 창의 끝
- θ: neutral-dev 12클립에서 가린 시간 비율이 5% 이하가 되는 가장 작은 값. 부록 1%
- 구간 recall: 확실한 구간 `[on, off]`에 대해 `[on, min(off, on + 3)]`과 창이 겹치는 패치 중 하나라도 `≥ θ`이면 잡음
- 탐지 지연 d = 처음 잡은 패치의 창 끝 − on + 렌더링 0.02 s
- 시간 recall: 확실한 구간 길이 중 가려진 길이 / 오탐: neutral-test 가린 시간 · 라벨 클립의 사이렌 구간(확실 · `?`) 밖 가린 시간
- 출처별 · 클립 부트스트랩 2000회(씨앗 20261123) 95% 구간
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .yamnet import HOP_S, MODEL, WIN_S

CLASSES = {390: "Siren", 317: "Police car (siren)", 318: "Ambulance (siren)",
           319: "Fire engine, fire truck (siren)"}
RENDER_S = 0.02
GRID = 0.01          # 가린 시간을 이 눈금으로 센다
SEED = 20261123
N_BOOT = 2000


# ── 점수 ──────────────────────────────────────────────────────────────

def peak(scores: np.ndarray) -> np.ndarray:
    """(패치 × 521) → 패치마다 사이렌 4개 클래스의 최댓값"""
    return scores[:, sorted(CLASSES)].max(axis=1)


def cmd_score(a) -> None:
    from .yamnet import Session
    sess = Session.load(Path(a.model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in a.wav:
        for w in sorted(Path(d).glob("*.wav")):
            np.save(out / f"{w.stem}.npy", peak(sess.scores(w)).astype(np.float32))
            n += 1
    print(f"점수 {n}클립 → {out}")


# ── 가림 · 지표 ───────────────────────────────────────────────────────

def covered_mask(p: np.ndarray, theta: float, dur: float) -> np.ndarray:
    """GRID 눈금마다 가렸는가 — 창 [0.48k, 0.48k+0.975)의 합집합, 클립 길이에서 자른다"""
    n = int(round(dur / GRID))
    m = np.zeros(n, bool)
    for k in np.nonzero(p >= theta)[0]:
        a = int(round(k * HOP_S / GRID))
        b = min(n, int(round((k * HOP_S + WIN_S) / GRID)))
        m[a:b] = True
    return m


def interval_mask(ivs, dur: float) -> np.ndarray:
    n = int(round(dur / GRID))
    m = np.zeros(n, bool)
    for on, off in ivs:
        m[int(round(on / GRID)):min(n, int(round(off / GRID)))] = True
    return m


def blur_frac(clips: dict[str, tuple[np.ndarray, float]], theta: float) -> float:
    cov = tot = 0
    for p, dur in clips.values():
        m = covered_mask(p, theta, dur)
        cov += int(m.sum())
        tot += len(m)
    return cov / tot if tot else float("nan")


def threshold_for(dev: dict[str, tuple[np.ndarray, float]], frac: float) -> float:
    """dev에서 가린 시간 비율이 frac 이하가 되는 가장 작은 θ (점수값 중에서, ≥ 판정)"""
    vals = np.unique(np.concatenate([p for p, _ in dev.values()]))
    lo, hi = 0, len(vals)                 # vals[hi:]는 모두 조건을 만족한다고 본다
    while lo < hi:
        mid = (lo + hi) // 2
        if blur_frac(dev, float(vals[mid])) <= frac:
            hi = mid
        else:
            lo = mid + 1
    return float(vals[lo]) if lo < len(vals) else float(np.nextafter(vals[-1], np.inf))


def first_catch(p: np.ndarray, theta: float, on: float, off: float) -> float | None:
    """[on, min(off, on+3)]과 창이 겹치는 첫 ≥θ 패치의 창 끝 — 없으면 None"""
    end = min(off, on + 3.0)
    for k in range(len(p)):
        s = k * HOP_S
        if s >= end:
            break
        if s + WIN_S > on and p[k] >= theta:
            return s + WIN_S
    return None


def per_clip(p: np.ndarray, dur: float, theta: float, pos, amb) -> dict:
    cov = covered_mask(p, theta, dur)
    posm = interval_mask(pos, dur)
    anym = posm | interval_mask(amb, dur)
    first = []
    for on, off in pos:
        c = first_catch(p, theta, on, off)
        first.append(None if c is None else c - on + RENDER_S)
    return dict(first=first, n=len(pos),
                cov_pos=int((cov & posm).sum()), tot_pos=int(posm.sum()),
                fp=int((cov & ~anym).sum()), neg=int((~anym).sum()))


def summarize(stats: list[dict]) -> dict:
    first = [f for s in stats for f in s["first"]]
    got = [f for f in first if f is not None]
    n = len(first)
    tp, tt = sum(s["cov_pos"] for s in stats), sum(s["tot_pos"] for s in stats)
    fp, ng = sum(s["fp"] for s in stats), sum(s["neg"] for s in stats)
    return dict(
        n_intervals=n, caught=len(got),
        interval_recall=len(got) / n if n else float("nan"),
        time_recall=tp / tt if tt else float("nan"),
        fp_outside=fp / ng if ng else float("nan"),
        latency_median=float(np.median(got)) if got else None,
        latency_p90=float(np.percentile(got, 90)) if got else None,
        latency_max=float(max(got)) if got else None,
    )


def bootstrap(per: dict[str, dict], rng) -> list[float] | None:
    keys = [k for k, s in per.items() if s["n"]]
    if len(keys) < 2:
        return None
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        st = [per[keys[i]] for i in pick]
        n = sum(s["n"] for s in st)
        vals.append(sum(f is not None for s in st for f in s["first"]) / n)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ── 평가 ──────────────────────────────────────────────────────────────

def _ids(path: str) -> list[str]:
    return [x.strip() for x in open(path, encoding="utf-8") if x.strip() and not x.startswith("#")]


def _dur(p: np.ndarray, sec: float | None) -> float:
    return sec if sec else (len(p) - 1) * HOP_S + WIN_S


def cmd_evaluate(a) -> None:
    from . import labels as L
    sd = Path(a.scores)
    load = lambda cid: np.load(sd / f"{cid}.npy")
    dev = {c: (load(c), _dur(load(c), None)) for c in _ids(a.neutral_dev)}
    test = {c: (load(c), _dur(load(c), None)) for c in _ids(a.neutral_test)}
    clips = list(csv.DictReader(open(a.clips, encoding="utf-8")))
    src = {r["clip_id"]: ("film" if r["source_class"] == "scare" else "real") for r in clips}
    durs = {r["clip_id"]: float(r["duration_s"]) for r in clips}
    lab = L.pick_annotator(L.load(a.labels), a.annotator)
    lab = [x for x in lab if x.category == "siren"]

    res = {"definitions": __doc__.split("**정의")[1].strip(), "classes": CLASSES}
    for name, frac in (("main_5pct", 0.05), ("appendix_1pct", 0.01)):
        th = threshold_for(dev, frac)
        per = {}
        for cid in src:
            p = load(cid)
            pos = [(x.onset, x.offset) for x in lab if x.clip_id == cid and not x.ambiguous]
            amb = [(x.onset, x.offset) for x in lab if x.clip_id == cid and x.ambiguous]
            per[cid] = per_clip(p, min(durs[cid], _dur(p, None)), th, pos, amb)
        out = dict(theta=th, dev_blur=blur_frac(dev, th), neutral_test_blur=blur_frac(test, th),
                   all=summarize(list(per.values())),
                   all_ci95=bootstrap(per, np.random.default_rng(SEED)), by_source={})
        for s in ("film", "real"):
            sub = {c: v for c, v in per.items() if src[c] == s}
            out["by_source"][s] = dict(**summarize(list(sub.values())),
                                       ci95=bootstrap(sub, np.random.default_rng(SEED)))
        out["per_clip"] = {c: dict(n=v["n"], first=v["first"]) for c, v in per.items()}
        res[name] = out
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    m = res["main_5pct"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print(f"θ {m['theta']:.4f} (dev {m['dev_blur']:.3f}) · neutral-test 가림 {m['neutral_test_blur']:.3f}")
    for k, v in [("전체", m["all"]), ("영화", m["by_source"]["film"]), ("실제", m["by_source"]["real"])]:
        print(f"{k}: 구간 {v['caught']}/{v['n_intervals']} recall {f(v['interval_recall'])} · "
              f"시간 recall {f(v['time_recall'])} · 구간 밖 가림 {f(v['fp_outside'])} · "
              f"지연 중앙값 {f(v['latency_median'])} p90 {f(v['latency_p90'])}")
    print(f"95% (전체) {m['all_ci95']} · (실제) {m['by_source']['real']['ci95']}")


# ── 자체 점검 ─────────────────────────────────────────────────────────

def cmd_selftest(_a=None) -> int:
    p = np.array([0.0, 0.9, 0.0, 0.0, 0.8])          # 창 [0.48,1.455) · [1.92,2.895)
    m = covered_mask(p, 0.5, 3.0)
    assert abs(m.sum() * GRID - (0.975 + 0.975)) < 0.02, m.sum()
    assert first_catch(p, 0.5, 1.0, 5.0) == HOP_S + WIN_S          # 창이 on 앞에서 시작해도 겹치면 잡음
    assert first_catch(p, 0.5, 1.5, 1.8) is None                   # [1.5,1.8]과 겹치는 ≥θ 창 없음
    assert first_catch(p, 0.5, 2.0, 9.0) == 4 * HOP_S + WIN_S
    dev = {"a": (np.array([0.1, 0.2, 0.3, 0.9]), 4 * 0.48 + 0.495)}
    th = threshold_for(dev, 0.30)
    assert blur_frac(dev, th) <= 0.30 and blur_frac(dev, np.nextafter(th, 0) if th > 0.1 else th) >= 0
    st = per_clip(p, 3.0, 0.5, [(1.0, 2.0)], [(2.5, 2.9)])
    assert st["first"][0] is not None and st["n"] == 1
    s = summarize([st])
    assert s["interval_recall"] == 1.0 and 0 < s["time_recall"] <= 1
    print("siren selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="사이렌 — YAMNet 기존 클래스 임계값 (#35 11.23)")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--wav", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default=str(MODEL))
    e = sub.add_parser("evaluate")
    for k in ("scores", "labels", "clips", "neutral-dev", "neutral-test", "out"):
        e.add_argument(f"--{k}", required=True)
    e.add_argument("--annotator", default="B")
    a = ap.parse_args(argv)
    if a.selftest:
        return cmd_selftest()
    if a.cmd == "score":
        cmd_score(a)
    elif a.cmd == "evaluate":
        cmd_evaluate(a)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
