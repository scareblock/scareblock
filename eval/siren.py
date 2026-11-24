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
# 사이렌 세트(#35 11.23) candidates.txt: 영화 검색 → scare, 실제 출동 검색 → neutral 로 잘랐다
SOURCE = {"scare": "film", "neutral": "real"}
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
    from .features import load_wav
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    durs = {}
    for d in a.wav:
        for w in sorted(Path(d).glob("*.wav")):
            np.save(out / f"{w.stem}.npy", peak(sess.scores(w)).astype(np.float32))
            x, sr = load_wav(w)
            durs[w.stem] = len(x) / sr          # 클립 길이는 점수 길이가 아니라 wav 길이로 (#62 리뷰 🟡3)
    (out / "_dur.json").write_text(json.dumps(durs, indent=1), encoding="utf-8")
    print(f"점수 {len(durs)}클립 → {out}")


# ── 가림 · 지표 ───────────────────────────────────────────────────────

def covered_mask(p: np.ndarray, theta: float, dur: float) -> np.ndarray:
    """GRID 눈금마다 가렸는가 — 창 [0.48k, 0.48k+0.975)의 합집합, 클립 길이에서 자른다"""
    n = int(round(dur / GRID))
    m = np.zeros(n, bool)
    w = int(np.ceil(WIN_S / GRID - 1e-9))   # 창 하나 = 칸 98개로 고정 (0.975/0.01 = 97.5 반올림이 갈리던 것, #62 리뷰)
    for k in np.nonzero(p >= theta)[0]:
        a = int(round(k * HOP_S / GRID))    # 0.48·k는 칸 경계에 정확히 떨어진다
        m[a:min(n, a + w)] = True
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
    """dev에서 가린 시간 비율이 frac 이하가 되는 가장 작은 θ (점수값 중에서, ≥ 판정).

    가림은 θ가 dev 점수값을 지날 때만 바뀌므로 조건을 만족하는 θ는 구간 `(v_{i−1}, v_i]`다.
    여기서는 그 **위쪽 끝 v_i**(실제 점수값)를 돌려준다 — 구간 안 어느 θ든 dev · 라벨 결과는 같고,
    처음 보는 점수에는 위쪽 끝이 보수적이다 (#62 리뷰). CLIP의 `threshold_for`와 같은 뜻이다."""
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


def boot_blur(clips: dict[str, tuple[np.ndarray, float]], theta: float, rng) -> list[float] | None:
    """neutral 가림 비율의 클립 부트스트랩 95% 구간"""
    keys = sorted(clips)
    if len(keys) < 2:
        return None
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals.append(blur_frac({str(i): clips[keys[k]] for i, k in enumerate(pick)}, theta))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def e3_row(lab, per: dict[str, dict]) -> dict:
    """E3 부록 사이렌 행 — `eval.e3`의 Event · summarize 그대로 (#35 11.23 결과의 E3 표를 이 코드로 재현한다)"""
    from .e3 import Event, summarize as e3sum
    ev, idx = [], {}
    for x in lab:
        if x.ambiguous or x.clip_id not in per:
            continue
        i = idx.get(x.clip_id, 0)
        idx[x.clip_id] = i + 1
        ev.append(Event("siren", x.clip_id, x.onset, x.offset, per[x.clip_id]["first"][i]))
    r = e3sum(ev)
    row = r["siren"]
    return dict(n=row["n"], caught=row["caught"], T_median=row["T"]["median"], T_over_3s=row["T_over_3s"],
                d_median=row["d_caught"]["median"] if row["d_caught"] else None,
                half_free_caught=row["half_free_caught"], half_free_caught_ci95=row["half_free_caught_ci95"],
                r0_median_caught=row["by_L"]["0"]["r_median_caught"],
                spearman_T_r0=r["_pooled"]["spearman_T_r0_caught"], by_L=row["by_L"])


# ── 평가 ──────────────────────────────────────────────────────────────

def _ids(path: str) -> list[str]:
    return [x.strip() for x in open(path, encoding="utf-8") if x.strip() and not x.startswith("#")]


def _dur(p: np.ndarray, sec: float | None) -> float:
    return sec if sec else (len(p) - 1) * HOP_S + WIN_S


def cmd_evaluate(a) -> None:
    from . import labels as L
    sd = Path(a.scores)
    load = lambda cid: np.load(sd / f"{cid}.npy")
    wavdur = json.loads((sd / "_dur.json").read_text()) if (sd / "_dur.json").exists() else {}
    dur = lambda cid: min(wavdur.get(cid, 1e9), _dur(load(cid), None))   # wav 길이 기준, 없으면 점수 길이
    dev_ids, test_ids = _ids(a.neutral_dev), _ids(a.neutral_test)
    dev = {c: (load(c), dur(c)) for c in dev_ids}
    test = {c: (load(c), dur(c)) for c in test_ids}
    clips = list(csv.DictReader(open(a.clips, encoding="utf-8")))
    # 출처는 명시적으로 — clipcut의 source_class는 scare | neutral뿐이다. 모르는 값이면 멈춘다 (#62 리뷰 🟡1)
    unknown = {r["source_class"] for r in clips} - set(SOURCE)
    if unknown:
        raise SystemExit(f"모르는 source_class {sorted(unknown)} — SOURCE에 출처를 적는다")
    # θ를 고른 dev · 오탐을 잰 test 클립은 라벨 지표에 넣지 않는다
    src = {r["clip_id"]: SOURCE[r["source_class"]] for r in clips
           if r["clip_id"] not in set(dev_ids) | set(test_ids)}
    durs = {cid: dur(cid) for cid in src}
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
            per[cid] = per_clip(p, durs[cid], th, pos, amb)
        out = dict(theta=th, dev_blur=blur_frac(dev, th), neutral_test_blur=blur_frac(test, th),
                   neutral_test_blur_ci95=boot_blur(test, th, np.random.default_rng(SEED)),
                   all=summarize(list(per.values())),
                   all_ci95=bootstrap(per, np.random.default_rng(SEED)), by_source={})
        for s in ("film", "real"):
            sub = {c: v for c, v in per.items() if src[c] == s}
            out["by_source"][s] = dict(**summarize(list(sub.values())),
                                       ci95=bootstrap(sub, np.random.default_rng(SEED)))
        out["per_clip"] = {c: dict(n=v["n"], first=v["first"]) for c, v in per.items()}
        out["e3_appendix"] = e3_row(lab, per)
        res[name] = out
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    m = res["main_5pct"]
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    print(f"θ {m['theta']:.4f} (dev {m['dev_blur']:.3f}) · neutral-test 가림 {m['neutral_test_blur']:.3f}")
    for k, v in [("전체", m["all"]), ("영화", m["by_source"]["film"]), ("실제", m["by_source"]["real"])]:
        print(f"{k}: 구간 {v['caught']}/{v['n_intervals']} recall {f(v['interval_recall'])} · "
              f"시간 recall {f(v['time_recall'])} · 구간 밖 가림 {f(v['fp_outside'])} · "
              f"지연 중앙값 {f(v['latency_median'])} p90 {f(v['latency_p90'])}")
    print(f"95% (전체) {m['all_ci95']} · (실제) {m['by_source']['real']['ci95']} · "
          f"neutral-test 가림 95% {m['neutral_test_blur_ci95']}")
    e = m["e3_appendix"]
    print(f"E3 부록: 사건 {e['n']} (잡음 {e['caught']}) · T 중앙값 {e['T_median']:.2f} s · T > 3 s {e['T_over_3s']:.2f} · "
          f"d 중앙값 {e['d_median']:.3f} s · 버퍼 없이 절반 넘게 가림 {e['half_free_caught']:.2f} · "
          f"노출 비율 중앙값 L=0 {e['r0_median_caught']:.2f}")


# ── 자체 점검 ─────────────────────────────────────────────────────────

def cmd_selftest(_a=None) -> int:
    p = np.array([0.0, 0.9, 0.0, 0.0, 0.8])          # 창 [0.48,1.455) · [1.92,2.895)
    m = covered_mask(p, 0.5, 3.0)
    assert abs(m.sum() * GRID - (0.975 + 0.975)) < 0.02, m.sum()
    assert first_catch(p, 0.5, 1.0, 5.0) == HOP_S + WIN_S          # 창이 on 앞에서 시작해도 겹치면 잡음
    assert first_catch(p, 0.5, 1.5, 1.8) is None                   # [1.5,1.8]과 겹치는 ≥θ 창 없음
    assert first_catch(p, 0.5, 2.0, 9.0) == 4 * HOP_S + WIN_S
    dev = {"a": (np.array([0.1, 0.2, 0.3, 0.9]), 4 * 0.48 + 0.495)}
    for frac in (0.30, 0.50, 0.80):
        th = threshold_for(dev, frac)
        vals = np.unique(dev["a"][0])
        assert blur_frac(dev, th) <= frac, (frac, th)
        below = vals[vals < th]
        if len(below):                       # 바로 아래 점수값에서는 넘어야 「가장 작은 값」이다
            assert blur_frac(dev, float(below[-1])) > frac, (frac, th, below[-1])
    assert int(covered_mask(np.ones(20), 0.5, 20.0)[:98].sum()) == 98      # 창 = 칸 98개로 고정
    assert all(covered_mask(np.eye(20)[k], 0.5, 20.0).sum() == 98 for k in range(15))
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
