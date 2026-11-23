/**
 * CLIP 시각 경로 — 04단계. 피 · 거미 · 주사기를 1초에 한 장씩 보고 가린다.
 *
 *   <video>(지연 전) ─ rVFC 1fps ─▶ 캔버스(원본 크기) ─ PIL과 같은 축소 · 자르기 ─▶ 224×224 RGB
 *     ─ InferClient.clip ─▶ offscreen: CLIP → 점수 ─▶ θ 넘으면 트리거 [t, t + 1/fps)
 *
 * 판정은 `eval/clip_zeroshot`의 정의 그대로다 — 프레임 t가 θ 이상이면 [t, t+1)을 가린다.
 * θ는 오프라인에서 neutral-dev 5%로 정한 값(#35)이고 config에 있다. 피는 보강 모델(#53)이 있으면 그것을 쓴다.
 *
 * 둘로 나눈다 (`visual.js` · `detector-rule.js`와 같다).
 *   SB.clipDecide   — 순수 계산. 점수 → 트리거. 브라우저 없이 테스트한다
 *   SB.ClipSampler  — <video>에 물려 프레임을 뽑고 offscreen에 보낸다
 *
 * **응답이 없으면(모델 없음 · 늦음) 가리지 않는다.** 이 경로는 원래 없던 것을 더하는 것이라
 * 실패하면 확장은 CLIP 이전과 똑같이 돈다.
 */
window.SB = window.SB || {};

/**
 * @param {{zs: object, probe: object}} r  offscreen 응답
 * @param {number} t                        프레임 미디어 시각
 * @param {object} cfg                      {CLIP_FPS, CLIP_THETA: {cat: θ}, CLIP_PROBE_THETA: {cat: θ}, CLIP_CATS}
 * @returns {Array} 트리거 — 계약(README 「인터페이스 계약」) 모양
 */
SB.clipDecide = (r, t, cfg = SB.config) => {
  const out = [];
  if (!r) return out;
  for (const cat of cfg.CLIP_CATS) {
    const useProbe = r.probe && typeof r.probe[cat] === 'number' && cfg.CLIP_PROBE_THETA?.[cat] != null;
    const p = useProbe ? r.probe[cat] : r.zs?.[cat];
    const theta = useProbe ? cfg.CLIP_PROBE_THETA[cat] : cfg.CLIP_THETA[cat];
    if (typeof p !== 'number' || theta == null) continue;
    if (p >= theta) {
      out.push({
        time: t,
        category: cat,
        confidence: p,
        duration: 1 / cfg.CLIP_FPS,
        source: useProbe ? 'clip-probe' : 'clip-zeroshot',
      });
    }
  }
  return out;
};

SB.ClipSampler = class {
  constructor(video, player, client, cfg = SB.config) {
    this.video = video;
    this.player = player;
    this.client = client;
    this.cfg = cfg;
    this.fps = cfg.CLIP_FPS;
    this.running = false;
    this.busy = false;          // 한 번에 하나만 보낸다 — 밀리면 그 프레임은 건너뛴다
    this.n = { 프레임: 0, 건너뜀: 0, 응답없음: 0, 늦음: 0, 트리거: 0 };
    this.byCat = {};
    this.prep = new SB.Samples();   // 페이지 쪽 축소 · 자르기 비용
    this.round = new SB.Samples();  // 보내고 받기까지
    this.lastError = null;
    this.failRun = 0;
    this.recent = [];
  }

  start() {
    this._makeCanvas();
    this._last = -1;
    this.running = true;
    this.video.requestVideoFrameCallback(this._onFrame);
    SB.log(`CLIP 시작 — ${this.fps}fps · ${this.cfg.CLIP_CATS.join(' · ')}`);
  }

  _makeCanvas() {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false });
  }

  _onFrame = (_now, meta) => {
    if (!this.running) return;
    const t = meta.mediaTime;
    if (t < this._last) this._last = -1;           // 뒤로 시크
    if (this._last < 0 || t - this._last >= 1 / this.fps - 0.002) {
      this._last = t;
      if (this.busy) this.n.건너뜀++;
      else {
        try { this._sample(t); this.failRun = 0; } catch (e) { this._fail(e); }
      }
    }
    if (this.running) this.video.requestVideoFrameCallback(this._onFrame);
  };

  _fail(e) {
    this.lastError = `${e.name}: ${e.message}`;
    this.failRun++;
    this._makeCanvas();                             // 다른 출처 프레임으로 오염된 캔버스는 버린다
    if (this.failRun >= 5) {
      this.running = false;
      SB.log(`CLIP 중단 — 5회 연속 실패 (${this.lastError})`);
    }
  }

  _sample(t) {
    const t0 = performance.now();
    // 원본 크기로 그린다 — eval도 원본 해상도 프레임을 PIL로 줄였다. 캔버스로 미리 줄이면 보간이 달라진다
    const w = this.video.videoWidth, h = this.video.videoHeight;
    if (!w || !h) return;
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.ctx.drawImage(this.video, 0, 0, w, h);
    const crop = SB.clipCrop(SB.rgbaToRgb(this.ctx.getImageData(0, 0, w, h).data), w, h);
    this.prep.push(performance.now() - t0);
    this.n.프레임++;
    this.busy = true;
    const t1 = performance.now();
    this.client.clip(crop).then((r) => {
      this.busy = false;
      if (!this.running) return;
      this.round.push(performance.now() - t1);
      if (!r) { this.n.응답없음++; return; }
      // 지연 버퍼 덕분에 대개 화면에 닿기 전에 온다. 이미 지나갔어도 넣는다 — 남은 구간은 가린다
      if (this.video.currentTime - this.cfg.DELAY_SEC > t + 1 / this.fps) this.n.늦음++;
      const trig = SB.clipDecide(r, t, this.cfg);
      this.recent.push({ t: +t.toFixed(1), ...Object.fromEntries(
        Object.entries({ ...r.zs, ...Object.fromEntries(Object.entries(r.probe || {}).map(([k, v]) => [`${k}_probe`, v])) })
          .map(([k, v]) => [k, +v.toFixed(3)])) });
      if (this.recent.length > 20) this.recent.shift();
      if (trig.length) {
        this.player.addTriggers(trig);
        this.n.트리거 += trig.length;
        for (const x of trig) this.byCat[x.category] = (this.byCat[x.category] || 0) + 1;
        SB.log(`CLIP ${trig.map((x) => `${x.category} ${x.confidence.toFixed(2)}`).join(' · ')} ` +
               `t=${t.toFixed(1)}s (지금 ${this.video.currentTime.toFixed(2)}s)`);
      }
    }, (e) => { this.busy = false; this.lastError = String(e?.message || e); });
  }

  report() {
    return {
      fps: this.fps,
      ...this.n,
      종류별: this.byCat,
      자르기_ms: this.prep.summary(),
      왕복_ms: this.round.summary(),
      ...(this.lastError ? { 마지막오류: this.lastError } : {}),
      ...(this.running ? {} : { 상태: '꺼짐' }),
      최근: this.recent,
    };
  }

  stop() { this.running = false; }
};
