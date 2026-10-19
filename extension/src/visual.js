/**
 * 시각 특징 — 03단계. 컷 전환 · 밝기 급변 · 전역 모션.
 *
 * README 「모델이 아닌 것」 그대로다 — 신경망이 아니라 프레임 간 산술이고,
 * **CLIP과 같은 1~2fps 샘플링에서 계산하는 것을 전제로** 비용이 거의 없다.
 * 720p 전 프레임에 돌리지 않는다. 64×36 휘도로 줄여서 본다.
 *
 * **아직 트리거를 내지 않는다.** 오디오·영상 융합은 향후 과제로 뺐고(README 03단계),
 * E1은 오디오 규칙만 잰다. 여기서는 특징을 계산해 기록하고 콘솔로 연다.
 * 이 기록이 04단계 CLIP 경로와 같은 샘플러를 쓰게 될 자리다.
 *
 * 둘로 나눈다 (`detector-rule.js`와 같다).
 *   SB.VisualCore    — 순수 계산. 휘도 배열을 받아 특징을 낸다. 브라우저 없이 테스트한다
 *   SB.VisualSampler — <video>에 rVFC로 물려 **지연 전** 프레임을 줄여 뽑는다
 */
window.SB = window.SB || {};

SB.VISUAL_PARAMS = {
  w: 64,
  h: 36,
  bins: 16,
  cut_hist: 0.40,      // 히스토그램 차이(반 L1, 0~1) 이상이면 컷
  flash_mean: 0.15,    // 평균 휘도 변화(0~1) 절댓값 이상이면 밝기 급변
};

SB.VisualCore = class {
  constructor(params = SB.VISUAL_PARAMS) {
    this.p = { ...SB.VISUAL_PARAMS, ...params };
    this.n = this.p.w * this.p.h;
    this.reset();
  }

  /** 시크·탭 복귀 뒤에는 직전 프레임과의 차이가 의미가 없다. 처음처럼 다시 시작한다. */
  reset() {
    this.prev = null;       // Uint8Array 휘도
    this.prevHist = null;   // Float32Array, 합 1
    this.prevMean = 0;
    this.prevT = -1;
  }

  /** RGBA(getImageData) → 휘도. Rec.601 정수 근사. */
  static luma(rgba, out) {
    for (let i = 0, j = 0; j < out.length; i += 4, j++) {
      out[j] = (77 * rgba[i] + 150 * rgba[i + 1] + 29 * rgba[i + 2]) >> 8;
    }
    return out;
  }

  /**
   * 휘도 한 장을 넣는다. 직전 프레임이 없으면 차이 특징은 null이다.
   * `maxGap` 초보다 벌어지거나 시각이 뒤로 가면 불연속으로 보고 차이를 내지 않는다.
   * @returns {{t, mean, dMean, hist, motion, cut, flash}}
   */
  push(y, t, maxGap = Infinity) {
    const { n, p } = this;
    const hist = new Float32Array(p.bins);
    const shift = 8 - Math.log2(p.bins);
    let sum = 0;
    for (let i = 0; i < n; i++) { sum += y[i]; hist[y[i] >> shift]++; }
    for (let b = 0; b < p.bins; b++) hist[b] /= n;
    const mean = sum / n / 255;

    if (this.prev && (t < this.prevT || t - this.prevT > maxGap)) this.reset();

    let dMean = null, histD = null, motion = null;
    if (this.prev) {
      dMean = mean - this.prevMean;
      histD = 0;
      for (let b = 0; b < p.bins; b++) histD += Math.abs(hist[b] - this.prevHist[b]);
      histD /= 2;
      // 전역 밝기 이동을 뺀 프레임 차이 — 페이드·플래시를 모션으로 세지 않는다
      const off = dMean * 255;
      let acc = 0;
      for (let i = 0; i < n; i++) acc += Math.abs(y[i] - this.prev[i] - off);
      motion = acc / n / 255;
    }

    if (!this.prev) this.prev = new Uint8Array(n);
    this.prev.set(y.subarray ? y.subarray(0, n) : y);
    this.prevHist = hist;
    this.prevMean = mean;
    this.prevT = t;

    return {
      t,
      mean,
      dMean,
      hist: histD,
      motion,
      cut: histD !== null && histD >= p.cut_hist,
      flash: dMean !== null && Math.abs(dMean) >= p.flash_mean,
    };
  }
};

/**
 * <video>에 물리는 쪽.
 *
 * 재생기와 **따로** rVFC를 건다. 재생기의 `_onFrame`은 M1 계측이 걸린 경로라 손대지 않는다 —
 * 한 영상에 rVFC 콜백은 여럿 걸 수 있다. 콜백은 매 프레임 오지만 `fps`마다 한 번만 뽑는다.
 *
 * `<video>`를 바로 그리므로 **지연 전** 프레임이다 — 탐지기가 미래를 먼저 본다(오디오와 같다).
 */
SB.VisualSampler = class {
  constructor(video, fps = SB.config.VISUAL_FPS, params = SB.VISUAL_PARAMS) {
    this.video = video;
    this.fps = fps;
    this.core = new SB.VisualCore(params);
    this.running = false;
    this.log = [];           // 최근 특징 — 링으로 자른다
    this.keep = 600;         // 2fps면 5분
    this.cuts = 0;
    this.flashes = 0;
    this.cost = new SB.Samples();
  }

  start() {
    const { w, h } = this.core.p;
    this.canvas = document.createElement('canvas');
    this.canvas.width = w;
    this.canvas.height = h;
    // 매번 읽어 가므로 CPU 쪽 캔버스로 둔다 — GPU 캔버스면 getImageData마다 되읽기가 생긴다
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
    this.y = new Uint8Array(w * h);
    this._last = -1;
    this.running = true;
    this.video.requestVideoFrameCallback(this._onFrame);
    SB.log(`시각 특징 시작 — ${this.fps}fps · ${w}x${h}`);
  }

  _onFrame = (_now, meta) => {
    if (!this.running) return;
    const t = meta.mediaTime;
    // 뒤로 시크하면 다음 샘플을 바로 뽑는다 — core가 불연속을 보고 차이를 비운다
    if (t < this._last) this._last = -1;
    if (this._last < 0 || t - this._last >= 1 / this.fps - 0.002) {
      this._last = t;
      this._sample(t);
    }
    this.video.requestVideoFrameCallback(this._onFrame);
  };

  _sample(t) {
    const t0 = performance.now();
    const { w, h } = this.core.p;
    this.ctx.drawImage(this.video, 0, 0, w, h);
    SB.VisualCore.luma(this.ctx.getImageData(0, 0, w, h).data, this.y);
    // 두 샘플 주기보다 벌어지면 불연속 — 탭 숨김(rVFC 정지)·시크가 여기 걸린다
    const f = this.core.push(this.y, t, 2.5 / this.fps);
    this.cost.push(performance.now() - t0);

    this.log.push(f);
    if (this.log.length > this.keep) this.log.splice(0, this.log.length - this.keep);
    if (f.cut) this.cuts++;
    if (f.flash) this.flashes++;
    if (f.cut || f.flash) {
      SB.log(
        `시각 ${f.cut ? '컷' : ''}${f.cut && f.flash ? '+' : ''}${f.flash ? '밝기급변' : ''} ` +
        `t=${t.toFixed(2)}s hist=${f.hist.toFixed(2)} dMean=${f.dMean.toFixed(2)} ` +
        `motion=${f.motion.toFixed(3)} (지금 ${this.video.currentTime.toFixed(2)}s)`
      );
    }
  }

  /** 콘솔용 요약. */
  report() {
    const recent = this.log.slice(-20).map((f) => ({
      t: +f.t.toFixed(2),
      mean: +f.mean.toFixed(3),
      dMean: f.dMean === null ? null : +f.dMean.toFixed(3),
      hist: f.hist === null ? null : +f.hist.toFixed(3),
      motion: f.motion === null ? null : +f.motion.toFixed(3),
      ...(f.cut ? { cut: true } : {}),
      ...(f.flash ? { flash: true } : {}),
    }));
    return {
      샘플fps: this.fps,
      샘플수: this.log.length,
      컷: this.cuts,
      밝기급변: this.flashes,
      샘플비용_ms: this.cost.summary(),
      최근: recent,
    };
  }

  stop() {
    this.running = false;
  }
};
