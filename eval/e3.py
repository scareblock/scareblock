"""E3 — 종류별 「얼마나 미리 알아야 하나」: 유지 시간 T와 필요 lookahead L의 관계.

제안서 §7 E3 · §3.2. 사전 등록은 #35 (11.22). 탐지기 · 임계값 · 라벨은 E1 · 04단계 그대로 쓴다.

정의 (등록문과 같다):
    T      = offset − onset
    d      = 첫 확신 시각 − onset + 렌더링 (규칙: E1 매칭의 fire / CLIP: 구간 안 첫 p ≥ θ 프레임)
    x(L)   = clip(d − L, 0, T)           가림이 시작되기 전에 보인 시간
    r(L)   = x(L) / T                    못 잡은 사건은 1
    「버퍼 없이 절반 넘게 가림」 = r(0) ≤ 0.5  ⇔  T ≥ 2d

사용:
    python3 -m eval.e3 --jump-labels B.csv --audio wav/ --clips clips-e1.csv \\
        --vis-labels V.csv V2.csv V3.csv --scores clip-scores/ --out e3.json
    python3 -m eval.e3 --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

LOOKAHEADS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
# E1 보고값(`e1 --render-s 0.02`, #35 B-rerun-main)과 같다. `evaluate.RENDER_S`(0.0, 미측정)와는 다르다 —
# M1 실측으로 E1을 다시 내면 `--render-s`도 같은 값으로 넘긴다 (#55 리뷰 3)
RENDER_S = 0.02
ANNOTATOR = "B"                     # E1 · 04단계 보고 주석자
SURGE_OF_QUIET = 20.0               # E1 보고 동작점
CATS = ("blood", "syringe", "spider")   # clip-scores npz 열 순서
THETA = {"blood": 0.2490658164024353,   # neutral-dev 5% (#35 04단계)
         "spider": 0.03803376853466034,
         "syringe": 0.0349690280854702}
SEED = 20261122
N_BOOT = 2000


@dataclass(frozen=True)
class Event:
    cat: str
    clip_id: str
    onset: float
    offset: float
    d: float | None                 # None = 못 잡음

    def __post_init__(self):
        # T로 나누므로 길이 0 · 음수 라벨은 받지 않는다 (#55 리뷰 2)
        if not self.offset > self.onset:
            raise ValueError(f"{self.clip_id} {self.cat} {self.onset}–{self.offset}: offset이 onset보다 앞서거나 같다")

    @property
    def T(self) -> float:
        return self.offset - self.onset

    def exposure(self, L: float) -> float:
        """노출 비율 r(L). 못 잡으면 1."""
        if self.d is None:
            return 1.0
        return float(min(max(self.d - L, 0.0), self.T) / self.T)


# ── 점프 스케어: 규칙 탐지기 ─────────────────────────────────────────

def jump_events(labels_csv: str, audio_dir: str, clips_csv: str,
                render_s: float = RENDER_S, annotator: str = ANNOTATOR) -> list[Event]:
    from . import labels as L
    from .detect import Params, run
    from .evaluate import match
    from .features import extract, load_wav

    truth = L.pick_annotator(L.load(labels_csv), annotator)   # E1 보고 주석자
    ids = L.load_clips(clips_csv)
    audio = {p.stem: p for p in Path(audio_dir).glob("*.wav")}
    used = [c for c in ids if c in audio]
    params = replace(Params(), surge_of_quiet=SURGE_OF_QUIET)
    truth = [t for t in truth if t.clip_id in used and t.category == "jumpscare"]
    dets = []
    for cid in used:
        x, sr = load_wav(audio[cid])
        dets += run(extract(x, sr), params, clip_id=cid)
    pairs, _, _ = match(truth, dets)
    fire = {id(t): d.fire for t, d in pairs}
    return [Event("jumpscare", t.clip_id, t.onset, t.offset,
                  (fire[id(t)] - t.onset + render_s) if id(t) in fire else None)
            for t in truth]


# ── 피 · 거미: CLIP zero-shot ───────────────────────────────────────

def clip_latency(p: np.ndarray, theta: float, on: float, off: float,
                 render_s: float = RENDER_S) -> float | None:
    """구간 [on, off] 안 1fps 프레임 중 처음 p ≥ θ인 프레임 → d. 없으면 None."""
    t = np.arange(len(p), dtype=float)
    ks = t[(t >= on) & (t <= off) & (p >= theta)]
    return float(ks[0] - on + render_s) if len(ks) else None


def vis_events(label_csvs: list[str], scores_dir: str, render_s: float = RENDER_S,
               annotator: str = ANNOTATOR) -> list[Event]:
    """시각 사건. `labels.load`로 읽고(ambiguous 파싱 · strip · 기본값이 E1과 같다) 주석자 하나만 쓴다 (#55 리뷰 1 · 4).

    두 사람이 같은 사건을 조금 다른 시각으로 찍은 이중 라벨을 같이 넘겨도 두 건으로 세지 않는다.
    """
    from . import labels as L
    out, seen, cache = [], set(), {}
    for f in label_csvs:
        allx = L.load(f)
        mine = L.pick_annotator(allx, annotator)
        # 주석자가 안 맞는 파일(열이 없거나 다른 사람)은 시각 사건이 조용히 0건이 된다 — 알린다 (#55 재리뷰 7)
        print(f"  {Path(f).name}: 라벨 {len(allx)}건 중 주석자 {annotator} {len(mine)}건", file=sys.stderr)
        if allx and not mine:
            print(f"  ⚠ {Path(f).name}에 주석자 {annotator}의 라벨이 없다 — 있는 것: {L.annotators(allx)}", file=sys.stderr)
        for x in mine:
            if x.category not in THETA or x.ambiguous:
                continue
            key = (x.clip_id, x.category, x.onset, x.offset)
            if key in seen:             # E1+V2 합본 파일이 겹쳐 들어와도 한 번만
                continue
            seen.add(key)
            if x.clip_id not in cache:
                z = np.load(Path(scores_dir) / f"{x.clip_id}.npz")
                assert tuple(z["cats"]) == CATS
                cache[x.clip_id] = z["p"]
            p = cache[x.clip_id][:, CATS.index(x.category)]
            out.append(Event(x.category, x.clip_id, x.onset, x.offset,
                             clip_latency(p, THETA[x.category], x.onset, x.offset, render_s)))
    return out


# ── 요약 ─────────────────────────────────────────────────────────────

def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic) if len(a) > 2 else float("nan")


def _q(xs):
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def _half_rate(evs: list[Event]) -> float:
    return float(np.mean([e.exposure(0.0) <= 0.5 for e in evs])) if evs else float("nan")


def _boot_half(evs: list[Event], rng) -> list[float] | None:
    """클립 단위 부트스트랩 — r(0) ≤ 0.5 비율의 95% 구간."""
    by: dict[str, list[Event]] = {}
    for e in evs:
        by.setdefault(e.clip_id, []).append(e)
    keys = sorted(by)
    if len(keys) < 2:
        return None
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = [e for k in pick for e in by[keys[k]]]
        vals.append(_half_rate(s))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(events: list[Event]) -> dict:
    res = {}
    for cat in sorted({e.cat for e in events}):
        ev = [e for e in events if e.cat == cat]
        got = [e for e in ev if e.d is not None]
        row = {
            "n": len(ev), "caught": len(got), "clips": len({e.clip_id for e in ev}),
            "T": _q([e.T for e in ev]),
            "T_over_3s": float(np.mean([e.T > 3.0 for e in ev])),
            "d_caught": _q([e.d for e in got]),
            # 라벨 onset이 늦게 찍혀 확신이 먼저 온 사건 — exposure에서 0(완전히 가림)으로 잘린다 (#55 리뷰 5)
            "d_negative_caught": int(sum(e.d < 0 for e in got)),
            "half_free_caught": _half_rate(got),
            "half_free_caught_ci95": _boot_half(got, np.random.default_rng(SEED)),
            "half_free_all": _half_rate(ev),
            "by_L": {},
        }
        for L in LOOKAHEADS:
            row["by_L"][f"{L:g}"] = {
                "r_median_caught": float(np.median([e.exposure(L) for e in got])) if got else None,
                "r_le_half_caught": float(np.mean([e.exposure(L) <= 0.5 for e in got])) if got else None,
                "zero_caught": float(np.mean([e.exposure(L) == 0.0 for e in got])) if got else None,
                "r_median_all": float(np.median([e.exposure(L) for e in ev])),
                "zero_all": float(np.mean([e.exposure(L) == 0.0 for e in ev])),
            }
        res[cat] = row
    got = [e for e in events if e.d is not None]
    res["_pooled"] = {"spearman_T_r0_caught": _spearman([e.T for e in got], [e.exposure(0) for e in got]),
                      "n_caught": len(got)}
    return res


def render(res: dict) -> str:
    cats = [c for c in res if not c.startswith("_")]
    f = lambda v, k=3: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{k}f}"
    lines = ["| 종류 | 사건 (잡음) | T 중앙값 | T > 3 s | d 중앙값 (잡음) | d < 0 | 버퍼 없이 절반 넘게 가림 (잡음) | 95% 구간 |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cats:
        r = res[c]
        ci = r["half_free_caught_ci95"]
        lines.append(f"| {c} | {r['n']} ({r['caught']}) | {f(r['T']['median'], 2)} s | {f(r['T_over_3s'], 2)} "
                     f"| {f(r['d_caught']['median'] if r['d_caught'] else None, 3)} s | {r['d_negative_caught']} "
                     f"| {f(r['half_free_caught'], 2)} | {'—' if ci is None else f'{ci[0]:.2f}–{ci[1]:.2f}'} |")
    lines += ["", "**L별 — 잡은 사건의 노출 비율 중앙값 / 노출 0 비율**", "",
              "| 종류 | " + " | ".join(f"L={L:g}" for L in LOOKAHEADS) + " |",
              "|---|" + "---|" * len(LOOKAHEADS)]
    for c in cats:
        cells = []
        for L in LOOKAHEADS:
            b = res[c]["by_L"][f"{L:g}"]
            cells.append(f"{f(b['r_median_caught'], 2)} / {f(b['zero_caught'], 2)}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    p = res["_pooled"]
    lines += ["", f"Spearman ρ(T, r(0)) 잡은 사건 {p['n_caught']}건 = {f(p['spearman_T_r0_caught'])}"]
    return "\n".join(lines)


# ── 자체 점검 ─────────────────────────────────────────────────────────

def _selftest() -> int:
    e = Event("x", "c", 10.0, 14.0, 1.0)          # T 4, d 1
    assert e.T == 4.0
    assert e.exposure(0) == 0.25 and e.exposure(0.5) == 0.125 and e.exposure(1) == 0.0
    assert e.exposure(5) == 0.0
    assert Event("x", "c", 0, 0.5, 2.0).exposure(0) == 1.0      # 끝난 뒤 잡음 → 다 보임
    assert Event("x", "c", 0, 1, None).exposure(3) == 1.0       # 못 잡음
    p = np.array([0, 0, 0, 0.1, 0.9, 0.9])
    assert clip_latency(p, 0.5, 2.5, 5.0, 0.0) == 1.5          # 프레임 4가 처음
    assert clip_latency(p, 0.5, 2.5, 3.5, 0.0) is None         # 구간 안 프레임 3은 낮음
    assert clip_latency(p, 0.5, 4.2, 4.8, 0.0) is None         # 구간 안에 프레임 없음 (1fps 한계)
    assert clip_latency(p, 0.5, 4.0, 5.0, 0.02) == 0.02
    evs = [Event("a", "c1", 0, 4, 1.0), Event("a", "c2", 0, 1, 0.8), Event("a", "c3", 0, 2, None),
           Event("b", "c4", 0, 0.4, 0.05)]
    s = summarize(evs)
    assert s["a"]["n"] == 3 and s["a"]["caught"] == 2
    assert s["a"]["half_free_caught"] == 0.5          # 1/4 ≤ .5 예, .8/1 아니오
    assert abs(s["a"]["half_free_all"] - 1 / 3) < 1e-12
    assert s["b"]["by_L"]["0"]["zero_caught"] == 0.0 and s["b"]["by_L"]["0.5"]["zero_caught"] == 1.0
    assert s["a"]["T_over_3s"] == 1 / 3
    assert s["b"]["d_negative_caught"] == 0
    assert summarize([Event("j", "c", 5, 6, -0.1)])["j"]["d_negative_caught"] == 1
    for on, off in ((3.0, 3.0), (3.0, 2.0)):          # 길이 0 · 음수 라벨은 받지 않는다
        try:
            Event("x", "c", on, off, None)
            raise AssertionError("받으면 안 된다")
        except ValueError:
            pass
    # 시각 라벨: 주석자 하나만 · ambiguous는 labels._to_bool 규칙(yes · 1 등) · 합본 중복은 한 번
    import tempfile
    with tempfile.TemporaryDirectory() as td:   # #55 재리뷰 8 — 끝나면 지운다
        d = Path(td)
        np.savez(d / "c1.npz", p=np.array([[0.0, 0, 0], [0.9, 0, 0], [0.9, 0, 0]]), cats=np.array(CATS))
        head = "video_id,clip_id,onset,offset,category,ambiguous,annotator\n"
        (d / "V.csv").write_text(head + "v,c1,0.5,2.5,blood,false,B\nv,c1,0.5,2.5,blood,yes,B\n"
                                 "v,c1,0.6,2.4,blood,false,A\n", encoding="utf-8")
        (d / "V2.csv").write_text(head + "v,c1,0.5,2.5,blood,false,B\n", encoding="utf-8")
        ev = vis_events([str(d / "V.csv"), str(d / "V2.csv")], str(d), render_s=0.0)
        assert [(e.clip_id, e.onset, e.d) for e in ev] == [("c1", 0.5, 0.5)], ev
    render(s)
    print("e3 selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E3 — 유지 시간 T와 필요 lookahead L")
    ap.add_argument("--jump-labels")
    ap.add_argument("--audio")
    ap.add_argument("--clips")
    ap.add_argument("--vis-labels", nargs="*", default=[])
    ap.add_argument("--scores")
    ap.add_argument("--out")
    ap.add_argument("--events-out", help="사건별 CSV (산점도용)")
    ap.add_argument("--render-s", type=float, default=RENDER_S,
                    help=f"렌더링 시간(초). E1을 낸 `e1 --render-s`와 같은 값 (기본 {RENDER_S})")
    ap.add_argument("--annotator", default=ANNOTATOR, help=f"이 주석자의 라벨만 (기본 {ANNOTATOR})")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not (a.jump_labels and a.audio and a.clips and a.vis_labels and a.scores):
        ap.error("--jump-labels --audio --clips --vis-labels --scores 가 모두 필요하다")
    ev = (jump_events(a.jump_labels, a.audio, a.clips, a.render_s, a.annotator)
          + vis_events(a.vis_labels, a.scores, a.render_s, a.annotator))
    print(f"주석자 {a.annotator} · 렌더링 {a.render_s:.2f} s 포함", file=sys.stderr)
    res = summarize(ev)
    print(render(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.events_out:
        with open(a.events_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["cat", "clip_id", "onset", "offset", "T", "d", "r0"])
            for e in ev:
                w.writerow([e.cat, e.clip_id, e.onset, e.offset, f"{e.T:.3f}",
                            "" if e.d is None else f"{e.d:.3f}", f"{e.exposure(0):.3f}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
