"""04단계 CLIP ViT-B/32 zero-shot 측정 (#35 11.05 등록 + 11.21 해시 댓글).

두 단계로 나눈다.

    # 1) 점수: 클립마다 1fps 프레임 → 종류별 확률을 .npz로 저장
    python -m eval.clip_zeroshot score --clips C.csv --video DIR --out DIR

    # 2) 평가: 기준값을 neutral-dev에서 정하고, 라벨 클립 · neutral-test에서 잰다
    python -m eval.clip_zeroshot evaluate --scores DIR --labels labels-V.csv \\
        --labeled-clips clips-V.csv --neutral-dev dev.txt --neutral-test test.txt

**정의 (결과를 보기 전에 고정)**

- 프레임: 클립 시작부터 1초마다 하나, `t = 0, 1, 2, …` 초. 원본 해상도로 디코드한 뒤
  CLIP 전처리(짧은 변 224 bicubic → 가운데 224 자르기 → mean/std)를 PIL로 한다.
  긴 변은 torchvision `Resize(224)`처럼 `int(224 · 긴 변 / 짧은 변)`으로 **버린다** (#51 리뷰 1)
- 프레임 `t`의 점수는 `[t, t+1)` 전체를 대표한다고 본다. 그래서 시간 recall은 `t < on`인 프레임도
  `[on, t+1)`을 가린 것으로 세고, 구간 recall · F1은 `t ≥ on`인 프레임만 센다(프레임 내용이 사건 이전이라)
- 점수 `p_c(t)` = 두 문장 softmax `[종류 문장, "an ordinary scene from a video"]`의 종류 쪽
  (logit_scale 100)
- 가림: `p_c(t) ≥ θ`인 프레임 `t`는 `[t, t+1)`을 가린다
- 기준값 `θ_c`: neutral-dev 프레임 중 가린 비율이 5% 이하가 되는 **가장 작은** 값.
  부록으로 1%도 낸다
- 정답: `ambiguous = false`인 구간만. `?` 구간은 정답에도 오탐에도 넣지 않는다
  (부록: `?`도 정답으로 넣은 값)
- 구간 recall: 구간 `[on, off]` 안에서 `on ≤ t ≤ min(off, on + 3)`인 프레임 하나라도
  `≥ θ`이면 잡음. 첫 발견 시간 = 처음 잡힌 `t − on`.
  기회 수는 onset의 소수부에 따라 3~4번이다(정수 onset이면 4번). 창 안에 정수 초가 없는 구간
  (1초 미만 · 클립 끝 부분 초)은 구조적으로 못 잡는다 — 결과에 `n_no_frame`으로 따로 낸다
- 시간 recall: 정답 구간 길이 중 가려진 길이의 비율 (전체 합)
- F1: 프레임 단위. 정답 프레임 = 확실한 구간 안의 `t`, `?` 구간 안의 프레임은 뺀다
- 오탐: neutral-test 가린 비율 · 라벨 클립에서 그 종류 구간(확실 · `?`) 밖의 가린 비율
- 이 숫자는 **모델 단독**이다. 확장의 블러 꼬리(1.5초)는 넣지 않는다
- 부트스트랩: 확실한 구간이 있는 클립을 복원 추출, 2000회, 씨앗 20261121, 95% 백분위
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path("_local/models/clip")
CATS = ("blood", "syringe", "spider")
SEED = 20261121
N_BOOT = 2000


# ── 1) 점수 ───────────────────────────────────────────────────────────────
def _probe(path: Path) -> dict:
    """크기(회전 반영) · 길이. ffmpeg는 기본으로 회전을 풀어 내보내므로 90°/270°면 w · h를 바꾼다 (#51 리뷰 2)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:stream_side_data=rotation:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if rot % 180:
        w, h = h, w
    vdur = float(s["duration"]) if s.get("duration") not in (None, "N/A") else None
    return dict(w=w, h=h, dur=float(j["format"]["duration"]), vdur=vdur, rot=rot)


class DecodeError(RuntimeError):
    pass


def frames_1fps(path: Path):
    """t = 0, 1, 2 … 초의 프레임을 RGB 배열로 낸다. `select`로 각 초 이후 첫 프레임을 고른다.

    다 내보낸 뒤 확인한다 — 틀리면 `DecodeError`라서 호출 쪽이 캐시하지 않는다 (#51 리뷰 3 · 4).
      · ffmpeg 반환 코드가 0이다
      · k번째로 고른 프레임의 pts가 정말 `floor(pts) == k`다 (VFR 빈 구간 · start_time ≥ 1이면 밀린다)
      · 프레임 수가 `floor(길이)`다. 영상 트랙만 짧은 클립(음성보다 먼저 끝남)은 경고만 하고 받는다
    """
    import tempfile
    info = _probe(path)
    w, h, dur = info["w"], info["h"], info["dur"]
    n = int(np.floor(dur))
    # 각 정수 초를 처음 넘는 프레임 하나씩: floor(t)가 바뀌는 프레임. showinfo로 고른 프레임의 pts를 받는다
    vf = "select='isnan(prev_selected_t)+gte(floor(t)-floor(prev_selected_t),1)',showinfo"
    with tempfile.TemporaryFile("w+") as err:
        p = subprocess.Popen(
            # showinfo는 info 수준으로 찍는다 — `-v error`면 pts가 안 나온다
            ["ffmpeg", "-nostats", "-v", "info", "-i", str(path), "-vf", vf, "-fps_mode", "vfr",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=err)
        size = w * h * 3
        k = 0
        try:
            while k < n:
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                yield k, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                k += 1
        finally:
            p.stdout.close()
            rc = p.wait()
        err.seek(0)
        log = err.read()
    if k < n and rc != 0:
        raise DecodeError(f"{path.name}: ffmpeg 반환 코드 {rc} — {log.strip()[-300:]}")
    pts = [float(x) for x in re.findall(r"Parsed_showinfo.*?pts_time:\s*([-0-9.]+)", log)][:k]
    if len(pts) != k:
        raise DecodeError(f"{path.name}: 고른 프레임 {k}장인데 pts는 {len(pts)}개")
    bad = [(i, t) for i, t in enumerate(pts) if int(np.floor(t)) != i]
    if bad:
        raise DecodeError(f"{path.name}: 프레임 번호와 시각이 어긋난다 (처음 {bad[0][0]}번째 = {bad[0][1]:.3f}s)")
    if k < n:
        vdur = info["vdur"]
        if vdur is not None and k >= int(np.floor(vdur)):
            print(f"  ⚠ {path.name}: 영상 트랙이 {vdur:.1f}s에서 끝난다 — 프레임 {k}장 (길이 {dur:.1f}s)",
                  file=sys.stderr)
        else:
            raise DecodeError(f"{path.name}: 프레임 {k}장 — floor(길이) {n}장보다 적다")


def resize_size(w: int, h: int, s: int = 224) -> tuple[int, int]:
    """torchvision `Resize(s)`와 같은 크기 — 짧은 변은 s, 긴 변은 `int(s · 긴 변 / 짧은 변)`(버림)."""
    if w <= h:
        return s, int(s * h / w)
    return int(s * w / h), s


def preprocess(rgb: np.ndarray, meta: dict) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(rgb)
    s = meta["size"]
    w, h = im.size
    im = im.resize(resize_size(w, h, s), Image.BICUBIC)
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
        try:
            p = score_clip(scorer, Path(a.video) / f"{cid}.mp4")
        except DecodeError as e:
            print(f"  ✗ {e} — 저장하지 않는다", file=sys.stderr)
            continue
        save_npz(dst, p=p, cats=np.array(CATS))
        print(f"{cid}: {len(p)} 프레임", file=sys.stderr)


def save_npz(dst: Path, **arrays) -> None:
    """임시 파일에 쓰고 rename — 중간에 끊겨도 반쪽 파일이 캐시로 남지 않는다 (#51 리뷰 4)."""
    tmp = dst.with_name(dst.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(dst)


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
    iv, no_frame = [], 0
    for on, off in pos:
        w = (t >= on) & (t <= min(off, on + 3.0))
        no_frame += int(not w.any())          # 창 안에 정수 초 프레임이 없다 — 구조적으로 못 잡는다
        ks = t[w & hit]
        iv.append(float(ks[0] - on) if len(ks) else None)
    cov, tot = _cover(t, hit, pos)
    pos_f = _inside(t, pos)
    amb_f = _inside(t, amb)
    use = ~amb_f
    tp = int((hit & pos_f & use).sum())
    fp_f = int((hit & ~pos_f & use).sum())
    fn = int((~hit & pos_f & use).sum())
    neg = use & ~pos_f                        # fp_f == (hit & neg).sum()
    return dict(first=iv, no_frame=no_frame, cov=cov, tot=tot, tp=tp, fp=fp_f, fn=fn,
                neg_n=int(neg.sum()))


def summarize(stats: list[dict]) -> dict:
    first = [x for s in stats for x in s["first"]]
    n = len(first)
    got = [x for x in first if x is not None]
    tp = sum(s["tp"] for s in stats); fp = sum(s["fp"] for s in stats); fn = sum(s["fn"] for s in stats)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    # 정답 프레임이 있으면 F1은 늘 정의된다 — 하나도 안 가렸어도(tp + fp = 0) 0이다 (#51 재리뷰 1).
    # 2·P·R/(P+R)와 같은 값이고, precision이 NaN인 경우까지 한 번에 처리한다
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else float("nan")
    cov = sum(s["cov"] for s in stats); tot = sum(s["tot"] for s in stats)
    nn = sum(s["neg_n"] for s in stats)
    return dict(
        n_intervals=n, n_no_frame=sum(s["no_frame"] for s in stats),
        interval_recall=(len(got) / n) if n else float("nan"),
        time_recall=(cov / tot) if tot else float("nan"),
        first_median=float(np.median(got)) if got else None,
        first_p90=float(np.percentile(got, 90)) if got else None,
        first_max=float(max(got)) if got else None,
        frame_precision=prec, frame_recall=rec, frame_f1=f1,
        fp_outside=(fp / nn) if nn else float("nan"))


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
    lab = [r["clip_id"] for r in clips]
    # 기준값은 dev에서 정하고 test에서 잰다 — 목록이 겹치면 test 오탐이 낙관적으로 나온다 (#51 리뷰 5)
    for x, y, nx, ny in ((dev, test, "neutral-dev", "neutral-test"), (dev, lab, "neutral-dev", "라벨 클립"),
                         (test, lab, "neutral-test", "라벨 클립")):
        both = sorted(set(x) & set(y))
        if both:   # assert가 아니다 — `python -O`에서도 꺼지면 안 된다 (#51 재리뷰 2)
            raise SystemExit(f"{nx}와 {ny}가 겹친다: {both}")

    cache: dict[str, np.ndarray] = {}
    def scores(cid: str) -> np.ndarray:
        if cid not in cache:
            cache[cid] = load_scores(sd, cid)
        return cache[cid]

    doc = __doc__ or ""
    result = {"definitions": doc.split("**정의")[1].strip() if "**정의" in doc else ""}
    for frac_name, frac in (("main_5pct", 0.05), ("appendix_1pct", 0.01)):
        block = {}
        for ci, cat in enumerate(CATS):
            devp = np.concatenate([scores(c)[:, ci] for c in dev])
            theta = threshold_for(devp, frac)
            testp = np.concatenate([scores(c)[:, ci] for c in test])
            out = dict(theta=theta, dev_blur=float((devp >= theta).mean()),
                       neutral_test_blur=float((testp >= theta).mean()))
            for variant, pos_amb in (("certain", False), ("with_ambiguous", None)):
                per = {}
                for r in clips:
                    cid = r["clip_id"]
                    p = scores(cid)[:, ci]
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
    txt = json.dumps(_nan_to_none(result), ensure_ascii=False, indent=1, default=float, allow_nan=False)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


def _nan_to_none(x):
    """표준 JSON에는 NaN이 없다 — 정의되지 않는 값은 null로 쓴다."""
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return None
    return x


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "CLIP zero-shot").split("\n")[0])
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
    # 정수 초를 안 지나는 구간 · 클립 끝 부분 초는 기회가 없다
    assert per_clip(q, 0.5, [(4.2, 4.8), (9.3, 9.9), (2.0, 5.0)], [])["no_frame"] == 2
    # tp = 0이면 F1은 NaN이 아니라 0
    z = per_clip(np.array([0.9, 0.0, 0.0]), 0.5, [(1.0, 2.0)], [])
    assert summarize([z])["frame_f1"] == 0.0
    # 하나도 안 가렸어도(tp + fp = 0) 정답이 있으면 0 — precision만 정의되지 않는다
    z0 = summarize([per_clip(np.zeros(3), 0.5, [(1.0, 2.0)], [])])
    assert z0["frame_f1"] == 0.0 and np.isnan(z0["frame_precision"])
    # 정답 프레임이 없으면 F1은 정의되지 않는다
    assert np.isnan(summarize([per_clip(np.ones(3), 0.5, [], [])])["frame_f1"])
    # 리사이즈 크기 — torchvision Resize(224)와 같다(버림). 16:9가 아닌 것에서 반올림과 갈린다
    for (w, h), want in {(1280, 720): (398, 224), (854, 480): (398, 224), (640, 480): (298, 224),
                         (480, 360): (298, 224), (1280, 536): (534, 224), (360, 640): (224, 398),
                         (224, 224): (224, 224)}.items():
        assert resize_size(w, h) == want, (w, h, resize_size(w, h))
    assert _nan_to_none({"a": [float("nan"), 1.0]}) == {"a": [None, 1.0]}
    print("selftest ok")


if __name__ == "__main__":
    main()
