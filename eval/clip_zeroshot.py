"""04단계 CLIP ViT-B/32 zero-shot 측정 (#35 11.05 등록 + 11.21 해시 댓글).

두 단계로 나눈다.

    # 1) 점수: 클립마다 1fps 프레임 → 종류별 확률을 .npz로 저장
    python -m eval.clip_zeroshot score --clips C.csv --video DIR --out DIR

    # 2) 평가: 기준값을 neutral-dev에서 정하고, 라벨 클립 · neutral-test에서 잰다
    python -m eval.clip_zeroshot evaluate --scores DIR --labels labels-V.csv \\
        --labeled-clips clips-V.csv --neutral-dev dev.txt --neutral-test test.txt

**정의 (결과를 보기 전에 고정)**

- 프레임: 클립 시작부터 1초마다 하나, `t = 0, 1, 2, …` 초. 원본 해상도로 디코드한 뒤
  CLIP 전처리(짧은 변 224 bicubic → 가운데 224 자르기 → mean/std)를 PIL로 한다
- 점수 `p_c(t)` = 두 문장 softmax `[종류 문장, "an ordinary scene from a video"]`의 종류 쪽
  (logit_scale 100)
- 가림: `p_c(t) ≥ θ`인 프레임 `t`는 `[t, t+1)`을 가린다
- 기준값 `θ_c`: neutral-dev 프레임 중 가린 비율이 5% 이하가 되는 **가장 작은** 값.
  부록으로 1%도 낸다
- 정답: `ambiguous = false`인 구간만. `?` 구간은 정답에도 오탐에도 넣지 않는다
  (부록: `?`도 정답으로 넣은 값)
- 구간 recall: 구간 `[on, off]` 안에서 `on ≤ t ≤ min(off, on + 3)`인 프레임 하나라도
  `≥ θ`이면 잡음. 첫 발견 시간 = 처음 잡힌 `t − on`
- 시간 recall: 정답 구간 길이 중 가려진 길이의 비율 (전체 합)
- F1: 프레임 단위. 정답 프레임 = 확실한 구간 안의 `t`, `?` 구간 안의 프레임은 뺀다
- 오탐: neutral-test 가린 비율 · 라벨 클립에서 그 종류 구간(확실 · `?`) 밖의 가린 비율
- 부트스트랩: 확실한 구간이 있는 클립을 복원 추출, 2000회, 씨앗 20261121, 95% 백분위
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path("_local/models/clip")
CATS = ("blood", "syringe", "spider")
SEED = 20261121
N_BOOT = 2000


# ── 1) 점수 ───────────────────────────────────────────────────────────────
def _probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j["format"]["duration"])


def frames_1fps(path: Path):
    """t = 0, 1, 2 … 초의 프레임을 RGB 배열로 낸다. `select`로 각 초 이후 첫 프레임을 고른다."""
    w, h, dur = _probe(path)
    n = int(np.floor(dur))
    # 각 정수 초를 처음 넘는 프레임 하나씩: floor(t)가 바뀌는 프레임
    vf = "select='isnan(prev_selected_t)+gte(floor(t)-floor(prev_selected_t),1)'"
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-fps_mode", "vfr",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    size = w * h * 3
    k = 0
    while k < n:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield k, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        k += 1
    p.stdout.close()
    p.wait()


def preprocess(rgb: np.ndarray, meta: dict) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(rgb)
    s = meta["size"]
    w, h = im.size
    scale = s / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = im.size
    left, top = int(round((w - s) / 2.0)), int(round((h - s) / 2.0))
    im = im.crop((left, top, left + s, top + s))
    x = np.asarray(im, np.float32) / 255.0
    x = (x - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    return x.transpose(2, 0, 1)


class Scorer:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.sess = ort.InferenceSession(str(model_dir / "visual.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.text = np.load(model_dir / "text_emb.npy").astype(np.float32)
        self.order = self.meta["order"]
        self.scale = float(self.meta["logit_scale"])

    def probs(self, batch: np.ndarray) -> np.ndarray:
        """(n, 3, 224, 224) → (n, len(CATS)) 종류 쪽 확률."""
        emb = self.sess.run(None, {"pixels": batch})[0]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        cos = emb @ self.text.T
        o = cos[:, self.order.index("ordinary")]
        out = []
        for c in CATS:
            d = self.scale * (cos[:, self.order.index(c)] - o)
            out.append(1.0 / (1.0 + np.exp(-d)))       # 2-way softmax = sigmoid(차이)
        return np.stack(out, 1)


def score_clip(scorer: Scorer, video: Path, batch: int = 32) -> np.ndarray:
    xs, res = [], []
    for _, rgb in frames_1fps(video):
        xs.append(preprocess(rgb, scorer.meta))
        if len(xs) == batch:
            res.append(scorer.probs(np.stack(xs)))
            xs = []
    if xs:
        res.append(scorer.probs(np.stack(xs)))
    return np.concatenate(res) if res else np.zeros((0, len(CATS)), np.float32)


def cmd_score(a) -> None:
    scorer = Scorer(Path(a.model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = [r["clip_id"] for r in csv.DictReader(open(a.clips))]
    if a.only:
        keep = set(Path(a.only).read_text().split())
        ids = [i for i in ids if i in keep]
    for cid in ids:
        dst = out / f"{cid}.npz"
        if dst.exists():
            continue
        p = score_clip(scorer, Path(a.video) / f"{cid}.mp4")
        np.savez(dst, p=p, cats=np.array(CATS))
        print(f"{cid}: {len(p)} 프레임", file=sys.stderr)


# ── 2) 평가 ───────────────────────────────────────────────────────────────
def load_scores(d: Path, cid: str) -> np.ndarray:
    z = np.load(d / f"{cid}.npz")
    assert tuple(z["cats"]) == CATS
    return z["p"]


def threshold_for(dev: np.ndarray, frac: float) -> float:
    """dev 프레임 중 `≥ θ` 비율이 frac 이하가 되는 가장 작은 θ."""
    s = np.sort(dev)[::-1]
    m = int(np.floor(frac * len(s)))           # 허용 프레임 수
    if m >= len(s):
        return float(s[-1])
    return float(np.nextafter(s[m], np.inf))   # s[m] 바로 위 → `≥ θ`인 것은 최대 m개


def intervals(labels: list[dict], cid: str, cat: str, amb: bool | None):
    out = []
    for r in labels:
        if r["clip_id"] != cid or r["category"] != cat:
            continue
        a = r["ambiguous"].strip().lower() == "true"
        if amb is None or a == amb:
            out.append((float(r["onset"]), float(r["offset"])))
    return out


def _inside(t: np.ndarray, ivs) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for on, off in ivs:
        m |= (t >= on) & (t <= off)
    return m


def _cover(t: np.ndarray, hit: np.ndarray, ivs) -> tuple[float, float]:
    """(가려진 길이, 전체 길이) — 가림은 [t, t+1)."""
    cov = tot = 0.0
    for on, off in ivs:
        tot += off - on
        for k in t[hit]:
            cov += max(0.0, min(off, k + 1) - max(on, k))
    return cov, tot


def per_clip(p: np.ndarray, theta: float, pos, amb):
    """한 클립 · 한 종류의 통계."""
    t = np.arange(len(p), dtype=float)
    hit = p >= theta
    iv = []
    for on, off in pos:
        w = (t >= on) & (t <= min(off, on + 3.0))
        ks = t[w & hit]
        iv.append(float(ks[0] - on) if len(ks) else None)
    cov, tot = _cover(t, hit, pos)
    pos_f = _inside(t, pos)
    amb_f = _inside(t, amb)
    use = ~amb_f
    tp = int((hit & pos_f & use).sum())
    fp_f = int((hit & ~pos_f & use).sum())
    fn = int((~hit & pos_f & use).sum())
    neg = use & ~pos_f
    return dict(first=iv, cov=cov, tot=tot, tp=tp, fp=fp_f, fn=fn,
                neg_hit=int((hit & neg).sum()), neg_n=int(neg.sum()))


def summarize(stats: list[dict]) -> dict:
    first = [x for s in stats for x in s["first"]]
    n = len(first)
    got = [x for x in first if x is not None]
    tp = sum(s["tp"] for s in stats); fp = sum(s["fp"] for s in stats); fn = sum(s["fn"] for s in stats)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not np.isnan(prec + rec) else float("nan")
    cov = sum(s["cov"] for s in stats); tot = sum(s["tot"] for s in stats)
    nh = sum(s["neg_hit"] for s in stats); nn = sum(s["neg_n"] for s in stats)
    return dict(
        n_intervals=n, interval_recall=(len(got) / n) if n else float("nan"),
        time_recall=(cov / tot) if tot else float("nan"),
        first_median=float(np.median(got)) if got else None,
        first_p90=float(np.percentile(got, 90)) if got else None,
        first_max=float(max(got)) if got else None,
        frame_precision=prec, frame_recall=rec, frame_f1=f1,
        fp_outside=(nh / nn) if nn else float("nan"))


def bootstrap(stats_by_clip: dict[str, dict], rng) -> tuple[float, float] | None:
    ids = [c for c, s in stats_by_clip.items() if s["first"]]
    if not ids:
        return None
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(ids), len(ids), replace=True)
        f = [x for i in pick for x in stats_by_clip[ids[i]]["first"]]
        vals.append(sum(x is not None for x in f) / len(f))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cmd_evaluate(a) -> None:
    sd = Path(a.scores)
    labels = list(csv.DictReader(open(a.labels)))
    clips = list(csv.DictReader(open(a.labeled_clips)))
    dev = Path(a.neutral_dev).read_text().split()
    test = Path(a.neutral_test).read_text().split()
    src = {r["clip_id"]: r["source"] for r in csv.DictReader(open(a.sources))} if a.sources else {}

    result = {"definitions": __doc__.split("**정의")[1].strip() if "**정의" in __doc__ else ""}
    for frac_name, frac in (("main_5pct", 0.05), ("appendix_1pct", 0.01)):
        block = {}
        for ci, cat in enumerate(CATS):
            devp = np.concatenate([load_scores(sd, c)[:, ci] for c in dev])
            theta = threshold_for(devp, frac)
            testp = np.concatenate([load_scores(sd, c)[:, ci] for c in test])
            out = dict(theta=theta, dev_blur=float((devp >= theta).mean()),
                       neutral_test_blur=float((testp >= theta).mean()))
            for variant, pos_amb in (("certain", False), ("with_ambiguous", None)):
                per = {}
                for r in clips:
                    cid = r["clip_id"]
                    p = load_scores(sd, cid)[:, ci]
                    if pos_amb is False:
                        pos = intervals(labels, cid, cat, False)
                        amb = intervals(labels, cid, cat, True)
                    else:
                        pos = intervals(labels, cid, cat, None)
                        amb = []
                    per[cid] = per_clip(p, theta, pos, amb)
                summ = summarize(list(per.values()))
                summ["interval_recall_ci95"] = bootstrap(per, np.random.default_rng(SEED))
                if src:
                    by = {}
                    for s_name in sorted({src.get(c, "") for c in per} - {""}):
                        sub = {c: v for c, v in per.items() if src.get(c) == s_name}
                        b = summarize(list(sub.values()))
                        b["interval_recall_ci95"] = bootstrap(sub, np.random.default_rng(SEED))
                        by[s_name] = b
                    summ["by_source"] = by
                out[variant] = summ
            block[cat] = out
        result[frac_name] = block
    txt = json.dumps(result, ensure_ascii=False, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--clips", required=True)
    s.add_argument("--video", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--only", help="이 목록(공백 구분)에 있는 클립만")
    s.add_argument("--model", default=str(MODEL_DIR))
    e = sub.add_parser("evaluate")
    e.add_argument("--scores", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--labeled-clips", required=True)
    e.add_argument("--neutral-dev", required=True)
    e.add_argument("--neutral-test", required=True)
    e.add_argument("--sources", help="clip_id,source CSV — 거미 출처별")
    e.add_argument("--out")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    {"score": cmd_score, "evaluate": cmd_evaluate, "selftest": cmd_selftest}[a.cmd](a)


def cmd_selftest(_a=None) -> None:
    """합성 점수로 기준값 · 지표 계산을 점검한다 (모델 · 영상 없이)."""
    d = np.arange(100) / 100.0
    th = threshold_for(d, 0.05)
    assert (d >= th).mean() <= 0.05 and (d >= np.nextafter(th, -np.inf) - 1e-9).mean() > 0.05
    p = np.zeros(20)
    p[5] = p[12] = 0.9
    # 구간 A [4.2, 8]: t=5에서 잡힘(0.8초) · 구간 B [10, 11.5]: 못 잡음 · `?` [15, 17]은 빼고 센다
    s = per_clip(p, 0.5, [(4.2, 8.0), (10.0, 11.5)], [(15.0, 17.0)])
    assert s["first"][1] is None and abs(s["first"][0] - 0.8) < 1e-9
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 5)
    assert abs(s["cov"] - 1.0) < 1e-9 and abs(s["tot"] - 5.3) < 1e-9
    m = summarize([s])
    assert m["interval_recall"] == 0.5 and abs(m["frame_f1"] - 0.25) < 1e-9
    assert s["neg_n"] == 11                     # 20 − 정답 6 − `?` 3
    # 3초 창: 구간 시작 3초 뒤에 처음 잡히면 넣고, 3초를 넘으면 뺀다
    q = np.zeros(10); q[4] = 1.0
    assert per_clip(q, 0.5, [(1.0, 9.0)], [])["first"] == [3.0]
    assert per_clip(q, 0.5, [(0.5, 9.0)], [])["first"] == [None]
    print("selftest ok")


if __name__ == "__main__":
    main()
