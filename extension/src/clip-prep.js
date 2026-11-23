/**
 * CLIP 전처리 — `eval/clip_zeroshot.py`(= OpenAI CLIP의 preprocess)와 **같은 값**을 내도록 옮긴 것.
 *
 *   짧은 변 224 · PIL BICUBIC(축소 시 안티에일리어싱) → 가운데 224×224 → /255 → mean/std
 *
 * 브라우저 캔버스로 줄이면 보간이 달라 점수가 흔들린다. 그러면 오프라인에서 정한 기준값(θ)이 그대로 맞지 않는다.
 * 그래서 PIL `Resample.c`의 8비트 경로(고정소수점 22비트 계수, 가로 → 세로, 패스마다 0~255로 자름)를 그대로 따른다.
 * 같은 RGB를 넣으면 PIL과 같은 바이트가 나오는지는 `test/clip-prep.test.js`가 본다.
 *
 * 페이지(줄이기 · 자르기)와 offscreen 문서(정규화 · 점수) 양쪽에서 쓴다. 입력은 RGB(알파 없음) Uint8Array.
 */
window.SB = window.SB || {};

SB.CLIP_SIZE = 224;
SB.CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
SB.CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

(() => {
  const PREC = 22;   // PIL PRECISION_BITS = 32 - 8 - 2

  function bicubic(x) {
    const a = -0.5;
    x = Math.abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
  }

  /** PIL precompute_coeffs + normalize_coeffs_8bpc. 출력 칸마다 [시작, 개수, 정수 계수들] */
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const fscale = Math.max(scale, 1);
    const support = 2.0 * fscale;
    const out = [];
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      const ss = 1 / fscale;
      // C의 (int) 변환은 0 쪽으로 자른다
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
      const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
      const k = new Float64Array(xmax);
      let ww = 0;
      for (let x = 0; x < xmax; x++) {
        const w = bicubic((x + xmin - center + 0.5) * ss);
        k[x] = w;
        ww += w;
      }
      const ki = new Int32Array(xmax);
      for (let x = 0; x < xmax; x++) {
        const v = ww !== 0 ? k[x] / ww : 0;
        ki[x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PREC)) : Math.trunc(0.5 + v * (1 << PREC));
      }
      out.push([xmin, xmax, ki]);
    }
    return out;
  }

  const clip8 = (ss) => {
    // PIL은 ss >> PREC를 0~255로 자른다. 산술 시프트는 음수에서도 내림이다
    const v = Math.floor(ss / (1 << PREC));
    return v < 0 ? 0 : v > 255 ? 255 : v;
  };

  /** RGB 가로 축소/확대 (채널 3) */
  function horiz(src, w, h, outW, cf) {
    const dst = new Uint8Array(outW * h * 3);
    for (let y = 0; y < h; y++) {
      const row = y * w * 3;
      for (let xx = 0; xx < outW; xx++) {
        const [xmin, n, k] = cf[xx];
        let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
        for (let x = 0; x < n; x++) {
          const p = row + (xmin + x) * 3;
          s0 += src[p] * k[x]; s1 += src[p + 1] * k[x]; s2 += src[p + 2] * k[x];
        }
        const o = (y * outW + xx) * 3;
        dst[o] = clip8(s0); dst[o + 1] = clip8(s1); dst[o + 2] = clip8(s2);
      }
    }
    return dst;
  }

  function vert(src, w, h, outH, cf) {
    const dst = new Uint8Array(w * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const [ymin, n, k] = cf[yy];
      for (let x = 0; x < w; x++) {
        let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
        for (let y = 0; y < n; y++) {
          const p = ((ymin + y) * w + x) * 3;
          s0 += src[p] * k[y]; s1 += src[p + 1] * k[y]; s2 += src[p + 2] * k[y];
        }
        const o = (yy * w + x) * 3;
        dst[o] = clip8(s0); dst[o + 1] = clip8(s1); dst[o + 2] = clip8(s2);
      }
    }
    return dst;
  }

  /** PIL Image.resize(BICUBIC) — 가로 먼저, 크기가 같으면 그 패스를 건너뛴다 (PIL과 같다) */
  SB.pilResize = (rgb, w, h, outW, outH) => {
    let cur = rgb, cw = w;
    if (outW !== w) { cur = horiz(cur, w, h, outW, coeffs(w, outW)); cw = outW; }
    if (outH !== h) cur = vert(cur, cw, h, outH, coeffs(h, outH));
    return cur;
  };

  /** 파이썬 round() — 반올림 동점은 짝수로 (torchvision CenterCrop이 이것을 쓴다) */
  const roundHalfEven = (v) => {
    const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  };

  /**
   * 짧은 변 224로 줄였을 때의 크기. `eval/clip_zeroshot.resize_size` = torchvision `Resize(224)`와 같다 —
   * 짧은 변은 정확히 224, 긴 변은 `int(224 · 긴 변 / 짧은 변)`(버림). 반올림하면 854×480(유튜브 480p)에서 1px 갈린다 (#51 리뷰 1)
   */
  SB.clipResizeSize = (w, h, s = SB.CLIP_SIZE) => (
    w <= h ? [s, Math.trunc((s * h) / w)] : [Math.trunc((s * w) / h), s]);

  /**
   * RGB(w×h) → 줄이고 가운데를 자른 RGB Uint8Array [224·224·3] (HWC).
   * 페이지에서 이것까지 하고 별도 문서로 보낸다 — 원본 프레임(720p면 2.7 MB)을 보내지 않으려고.
   */
  SB.clipCrop = (rgb, w, h) => {
    const s = SB.CLIP_SIZE;
    const [rw, rh] = SB.clipResizeSize(w, h, s);
    const r = SB.pilResize(rgb, w, h, rw, rh);
    const left = roundHalfEven((rw - s) / 2), top = roundHalfEven((rh - s) / 2);
    const out = new Uint8Array(s * s * 3);
    for (let y = 0; y < s; y++) {
      out.set(r.subarray(((top + y) * rw + left) * 3, ((top + y) * rw + left + s) * 3), y * s * 3);
    }
    return out;
  };

  /** 자른 RGB [224·224·3] → Float32Array [3, 224, 224] (/255 → mean/std) */
  SB.clipNormalize = (crop) => {
    const s = SB.CLIP_SIZE, n = s * s;
    if (crop.length !== n * 3) throw new Error(`CLIP 입력 ${crop.length}바이트 — ${n * 3}이어야 한다`);
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) out[c * n + i] = (crop[i * 3 + c] / 255 - SB.CLIP_MEAN[c]) / SB.CLIP_STD[c];
    }
    return out;
  };

  /** RGB(w×h) → Float32Array [3, 224, 224] */
  SB.clipPreprocess = (rgb, w, h) => SB.clipNormalize(SB.clipCrop(rgb, w, h));

  /** getImageData의 RGBA → RGB */
  SB.rgbaToRgb = (rgba, out = new Uint8Array((rgba.length / 4) * 3)) => {
    for (let i = 0, j = 0; j < out.length; i += 4, j += 3) {
      out[j] = rgba[i]; out[j + 1] = rgba[i + 1]; out[j + 2] = rgba[i + 2];
    }
    return out;
  };

  /**
   * 이미지 임베딩 → 종류별 점수. `eval/clip_zeroshot.Scorer.probs` · `eval/blood_probe score`와 같은 식.
   *   zero-shot: sigmoid(logit_scale · (cos_c − cos_ordinary))   (두 문장 softmax와 같다)
   *   보강 모델: sigmoid(w · e + b)                                 (e는 L2 정규화한 임베딩)
   * heads = {order, logit_scale, text_emb: [[512]…], probe?: {cat, coef, intercept}}
   * @returns {{zs: {cat: p}, probe: {cat: p}}}
   */
  SB.clipHeads = (emb, heads) => {
    let nn = 0;
    for (let i = 0; i < emb.length; i++) nn += emb[i] * emb[i];
    nn = Math.sqrt(nn);
    const dot = (v) => { let s = 0; for (let i = 0; i < emb.length; i++) s += (emb[i] / nn) * v[i]; return s; };
    const cos = heads.text_emb.map(dot);
    const o = cos[heads.order.indexOf('ordinary')];
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const zs = {};
    heads.order.forEach((c, i) => { if (c !== 'ordinary') zs[c] = sig(heads.logit_scale * (cos[i] - o)); });
    const probe = {};
    if (heads.probe) probe[heads.probe.cat] = sig(dot(heads.probe.coef) + heads.probe.intercept);
    return { zs, probe };
  };
})();
