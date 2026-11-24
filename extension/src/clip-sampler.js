/**
 * CLIP 시각 경로 — 04단계. 피 · 거미 · 주사기를 1초에 한 장씩 보고 가린다.
 *
 *   <video>(지연 전) ─ rVFC 1fps ─ createImageBitmap ─▶ 확장 프레임(clip-frame.js, 다른 프로세스):
 *     원본 크기 그리기 · PIL과 같은 축소 · 자르기 ─▶ offscreen: CLIP → 점수 ─▶ θ 넘으면 트리거 [t, t + 1/fps)
 *
 * 자르기를 페이지에서 하면 1080p에서 메인 스레드 66 ms라 1초마다 렌더가 끊겼다(11.24 실기) — 그래서 프레임으로 옮겼다
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
    this.prep = new SB.Samples();   // 페이지 쪽 비용 — 이제 createImageBitmap뿐
    this.frameCrop = new SB.Samples();   // 확장 프레임 안 그리기 · 읽기 · 자르기 (다른 프로세스)
    this.round = new SB.Samples();  // 보내고 받기까지
    this.lastError = null;
    this.failRun = 0;
    this.recent = [];
  }

  start() {
    this._last = -1;
    this.running = true;
    this.video.requestVideoFrameCallback(this._onFrame);
    SB.log(`CLIP 시작 — ${this.fps}fps · ${this.cfg.CLIP_CATS.join(' · ')} · 자르기는 확장 프레임에서`);
  }

  _onFrame = (_now, meta) => {
    if (!this.running) return;
    const t = meta.mediaTime;
    if (t < this._last) this._last = -1;           // 뒤로 시크
    if (this._last < 0 || t - this._last >= 1 / this.fps - 0.002) {
      this._last = t;
      if (this.busy) this.n.건너뜀++;
      else this._sample(t).then(() => { this.failRun = 0; }, (e) => this._fail(e));
    }
    if (this.running) this.video.requestVideoFrameCallback(this._onFrame);
  };

  _fail(e) {
    this.busy = false;
    this.lastError = `${e?.name || 'Error'}: ${e?.message || e}`;
    this.failRun++;
    if (this.failRun >= 5) {
      this.running = false;
      SB.log(`CLIP 중단 — 5회 연속 실패 (${this.lastError})`);
    }
  }

  async _sample(t) {
    if (!this.video.videoWidth || !this.video.videoHeight) return;
    this.busy = true;
    const t0 = performance.now();
    // 페이지에서는 프레임을 떠서 넘기기만 한다(0.3 ms). 원본 크기 그대로라 eval의 PIL 축소와 바이트가 같다
    const bitmap = await createImageBitmap(this.video);
    this.prep.push(performance.now() - t0);
    this.n.프레임++;
    const t1 = performance.now();
    const r = await this.client.clip(bitmap);
    this.busy = false;
    if (!this.running) return;
    this.round.push(performance.now() - t1);
    if (!r) {
      this.n.응답없음++;
      if (this.client.lastError) this.lastError = this.client.lastError;
      return;
    }
    if (typeof r.cropMs === 'number') this.frameCrop.push(r.cropMs);
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
  }

  report() {
    return {
      fps: this.fps,
      ...this.n,
      종류별: this.byCat,
      페이지_ms: this.prep.summary(),
      프레임_자르기_ms: this.frameCrop.summary(),
      왕복_ms: this.round.summary(),
      ...(this.lastError ? { 마지막오류: this.lastError } : {}),
      ...(this.running ? {} : { 상태: '꺼짐' }),
      최근: this.recent,
    };
  }

  stop() { this.running = false; }
};

/**
 * 페이지 → 확장 자르기 프레임(clip-frame.js). 프레임은 중계(bridge.js)가 달라고 할 때 넣는다.
 * 실패 · 시간 초과는 null — 샘플러는 그 장을 가리지 않는다(지금까지와 같은 방향).
 */
SB.ClipFrameClient = class {
  constructor(win = window, doc = document, timeoutMs = 3000) {
    this.win = win;
    this.doc = doc;
    this.timeoutMs = timeoutMs;
    this.seq = 0;
    this.pending = new Map();
    this.frame = null;
    this.lastError = null;
    this._onMsg = (e) => {
      if (!this.frame || e.source !== this.frame.contentWindow || e.data?.tag !== 'sb-clipframe') return;
      const p = this.pending.get(e.data.id);
      if (!p) return;
      this.pending.delete(e.data.id);
      clearTimeout(p.timer);
      p.resolve(e.data);
    };
    this.win.addEventListener('message', this._onMsg);
  }

  /** 프레임을 달라고 하고, 붙어서 대답할 때까지 기다린다. @returns {Promise<boolean>} */
  async ready(waitMs = 8000) {
    const until = performance.now() + waitMs;
    this.win.postMessage({ tag: 'sb-clipframe', op: 'need' }, this.win.location.origin);
    while (performance.now() < until) {
      this.frame = this.doc.getElementById('scareblock-clip-frame');
      if (this.frame?.contentWindow) {
        const r = await this._call('ping', {}, [], 500);
        if (r?.ok) return true;
      }
      await new Promise((r) => setTimeout(r, 200));
    }
    this.lastError = this.frame ? '프레임이 대답하지 않는다' : '프레임이 안 붙었다(bridge.js)';
    return false;
  }

  _call(op, payload, transfer, timeoutMs = this.timeoutMs) {
    const id = ++this.seq;
    return new Promise((resolve) => {
      const timer = setTimeout(() => { this.pending.delete(id); resolve(null); }, timeoutMs);
      this.pending.set(id, { resolve, timer });
      // new URL('chrome-extension://…').origin은 환경에 따라 'null'이다 — 문자열에서 바로 뗀다
      const origin = this.frame.src.match(/^[a-z][a-z0-9+.-]*:\/\/[^/]+/i)[0];
      this.frame.contentWindow.postMessage({ tag: 'sb-clipframe', op, id, ...payload }, origin, transfer);
    });
  }

  /** ImageBitmap(옮겨 간다) → {zs, probe, ms, cropMs} 또는 null */
  async clip(bitmap) {
    if (!this.frame) { bitmap.close?.(); return null; }
    const r = await this._call('clip', { bitmap }, [bitmap]);
    if (!r) { this.lastError = `${this.timeoutMs} ms 안에 프레임 응답 없음`; return null; }
    if (!r.ok) { this.lastError = r.error; return null; }
    return { ...r.result, cropMs: r.cropMs };
  }

  dispose() {
    this.win.removeEventListener('message', this._onMsg);
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.resolve(null); }
    this.pending.clear();
  }
};
