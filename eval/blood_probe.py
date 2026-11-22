"""유혈 보강 모델 — CLIP 이미지 임베딩 위 로지스틱 회귀 (#35 11.22 등록).

    # 1) 임베딩: 클립마다 1fps 512차원 임베딩(정규화)을 저장
    python -m eval.blood_probe embed --clips C.csv --video DIR --out EMB_DIR

    # 2) 학습: 라벨한 클립으로 학습, C는 클립 단위 교차 검증(평균 로그 손실)으로 고른다
    python -m eval.blood_probe train --emb EMB_DIR --labels L.csv [L2.csv …] \\
        --clips C.csv [C2.csv …] --out probe.json

    # 3) 점수: 학습한 모델로 클립마다 유혈 확률 → clip_zeroshot과 같은 .npz (blood 열만 바뀜)
    python -m eval.blood_probe score --emb EMB_DIR --probe probe.json --zs ZS_DIR --out OUT_DIR

평가는 `eval.clip_zeroshot evaluate`를 그대로 쓴다 (같은 기준값 규칙 · 같은 지표).

**정의 (등록 그대로)**
- 양성 = 확실한 유혈 구간 안 프레임 (`on ≤ t ≤ off`), `?` 유혈 구간 안 프레임은 뺀다. 나머지는 음성
- 특징 = 정규화한 512차원 임베딩. 표준화하지 않는다
- C 후보 0.01 · 0.1 · 1 · 10, 클래스 가중치 balanced, 클립 단위 5겹 교차 검증 · 평균 로그 손실 최소
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .clip_zeroshot import CATS, MODEL_DIR, Scorer, frames_1fps, preprocess

C_GRID = (0.01, 0.1, 1.0, 10.0)
N_FOLDS = 5
SEED = 20261122


def cmd_embed(a) -> None:
    sc = Scorer(Path(a.model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(a.clips)):
        cid = r["clip_id"]
        dst = out / f"{cid}.npy"
        if dst.exists():
            continue
        xs, embs = [], []
        for _, rgb in frames_1fps(Path(a.video) / f"{cid}.mp4"):
            xs.append(preprocess(rgb, sc.meta))
            if len(xs) == 32:
                embs.append(sc.sess.run(None, {"pixels": np.stack(xs)})[0]); xs = []
        if xs:
            embs.append(sc.sess.run(None, {"pixels": np.stack(xs)})[0])
        e = np.concatenate(embs).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        np.save(dst, e)
        print(f"{cid}: {len(e)} 프레임", file=sys.stderr)


def frame_labels(n: int, labels: list[dict], cid: str) -> np.ndarray:
    """1 = 유혈, 0 = 아님, -1 = `?` 유혈 구간(뺀다)."""
    t = np.arange(n, dtype=float)
    y = np.zeros(n, int)
    for r in labels:
        if r["clip_id"] != cid or r["category"] != "blood":
            continue
        on, off = float(r["onset"]), float(r["offset"])
        m = (t >= on) & (t <= off)
        amb = r["ambiguous"].strip().lower() == "true"
        y[m & (y != 1)] = -1 if amb else 1
    return y


def load_train(emb: Path, label_files, clip_files):
    labels = [r for f in label_files for r in csv.DictReader(open(f))]
    X, Y, G = [], [], []
    for f in clip_files:
        for r in csv.DictReader(open(f)):
            cid = r["clip_id"]
            e = np.load(emb / f"{cid}.npy")
            y = frame_labels(len(e), labels, cid)
            keep = y >= 0
            X.append(e[keep]); Y.append(y[keep]); G += [cid] * int(keep.sum())
    return np.concatenate(X), np.concatenate(Y), np.array(G)


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X, y)


def cmd_train(a) -> None:
    from sklearn.metrics import log_loss
    X, y, g = load_train(Path(a.emb), a.labels, a.clips)
    clips = sorted(set(g))
    pos_clips = sorted({c for c in clips if y[g == c].any()})
    rng = np.random.default_rng(SEED)
    # 양성 클립이 겹마다 흩어지도록: 양성 클립 · 나머지를 따로 섞어 번갈아 배정
    order = list(rng.permutation(pos_clips)) + list(rng.permutation([c for c in clips if c not in pos_clips]))
    fold = {c: i % N_FOLDS for i, c in enumerate(order)}
    f = np.array([fold[c] for c in g])
    cv = {}
    for C in C_GRID:
        losses = []
        for k in range(N_FOLDS):
            tr, te = f != k, f == k
            if y[tr].min() == y[tr].max() or not te.any():
                continue
            m = _fit(X[tr], y[tr], C)
            losses.append(log_loss(y[te], m.predict_proba(X[te])[:, 1], labels=[0, 1]))
        cv[C] = float(np.mean(losses))
    best = min(C_GRID, key=lambda c: (cv[c], c))
    m = _fit(X, y, best)
    out = dict(C=best, cv_logloss=cv, coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]),
               n_frames=int(len(y)), n_pos=int(y.sum()), n_clips=len(clips), pos_clips=pos_clips,
               folds={c: int(k) for c, k in fold.items()})
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("coef", "folds")}, ensure_ascii=False))


def cmd_score(a) -> None:
    pr = json.loads(Path(a.probe).read_text())
    w, b = np.array(pr["coef"], np.float32), pr["intercept"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.emb).glob("*.npy")):
        cid = f.stem
        e = np.load(f)
        p = 1.0 / (1.0 + np.exp(-(e @ w + b)))
        zs = np.load(Path(a.zs) / f"{cid}.npz")["p"].copy()
        assert len(zs) == len(p), cid
        zs[:, CATS.index("blood")] = p
        np.savez(out / f"{cid}.npz", p=zs, cats=np.array(CATS))


def cmd_selftest(_a=None) -> None:
    lab = [dict(clip_id="x", category="blood", onset="2", offset="4", ambiguous="false"),
           dict(clip_id="x", category="blood", onset="6.5", offset="8", ambiguous="true"),
           dict(clip_id="x", category="spider", onset="0", offset="9", ambiguous="false")]
    y = frame_labels(10, lab, "x")
    assert y.tolist() == [0, 0, 1, 1, 1, 0, 0, -1, -1, 0], y
    print("selftest ok")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--clips", required=True); e.add_argument("--video", required=True)
    e.add_argument("--out", required=True); e.add_argument("--model", default=str(MODEL_DIR))
    t = sub.add_parser("train")
    t.add_argument("--emb", required=True); t.add_argument("--labels", nargs="+", required=True)
    t.add_argument("--clips", nargs="+", required=True); t.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--emb", required=True); s.add_argument("--probe", required=True)
    s.add_argument("--zs", required=True); s.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    {"embed": cmd_embed, "train": cmd_train, "score": cmd_score, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
