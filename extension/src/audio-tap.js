/**
 * 지연 **전** 오디오를 16 kHz로 줄여 최근 몇 초를 들고 있는다 — YAMNet 패치를 미디어 시각으로 자르기 위해.
 *
 * 규칙 탐지기와 같은 자리(재생기의 소스)에 따로 물린다. 계산은 `pcm.js`의 SB.TapCore이고,
 * 여기는 오디오 그래프와 원점 관리만 한다. 원점을 다시 잡는 조건은 `detector-rule.js`와 같다 —
 * 두 곳이 다른 원점을 쓰면 탐지 시각과 패치가 어긋난다.
 */
window.SB = window.SB || {};

SB.AudioTap = class {
  constructor(video, player, keepS = 8) {
    this.video = video;
    this.player = player;
    this.keepS = keepS;
    this.waiters = [];
    this.costMs = 0;      // 콜백에서 쓴 시간 합 — M1 재측정 때 같이 본다
    this.calls = 0;
  }

  start() {
    const { ac, src } = this.player.audio;
    this.core = new SB.TapCore(ac.sampleRate, this.keepS);
    this.node = ac.createScriptProcessor(2048, 2, 1);
    this.mono = new Float32Array(2048);

    this.node.onaudioprocess = (e) => {
      if (this.video.paused || this.video.seeking) return;
      const t0 = performance.now();
      const inp = e.inputBuffer;
      const a = inp.getChannelData(0);
      const b = inp.numberOfChannels > 1 ? inp.getChannelData(1) : a;
      for (let k = 0; k < a.length; k++) this.mono[k] = (a[k] + b[k]) / 2;
      const rate = this.video.playbackRate || 1;
      if (this.core.anchor === null) {
        this.core.setAnchor(this.video.currentTime - inp.duration * rate, rate);
      }
      this.core.push(this.mono.subarray(0, a.length));
      this._wake();
      this.costMs += performance.now() - t0;
      this.calls++;
    };

    this._onReset = () => { this.core.reset(); this._stalled = false; this._wake(); };
    this._onWaiting = () => { this._stalled = true; };
    this._onPlaying = () => { if (this._stalled) this._onReset(); };
    this._listeners = [
      ['seeked', this._onReset], ['ratechange', this._onReset],
      ['waiting', this._onWaiting], ['playing', this._onPlaying],
    ];
    for (const [ev, fn] of this._listeners) this.video.addEventListener(ev, fn);

    src.connect(this.node);
    this.node.connect(ac.destination);   // 이어야 콜백이 돈다. 출력은 무음
    SB.log(`오디오 탭 시작 — ${ac.sampleRate} Hz → 16 kHz, 최근 ${this.keepS}초`);
  }

  /**
   * 미디어 시각 `start`부터 패치 하나(오디오 0.975초)를 기다려 받는다.
   * 그 구간이 고리에서 밀려났거나, 기다리는 동안 원점이 바뀌었거나(시크 등), 시간 안에 안 들어오면 null.
   */
  patchAt(start, timeoutMs = 4000) {
    const s = this.core.slice(start);
    if (s.pcm) return Promise.resolve(s.pcm);
    if (s.gone) return Promise.resolve(null);
    return new Promise((resolve) => {
      const w = { start, gen: this.core.gen, resolve };
      w.timer = setTimeout(() => this._settle(w, null), timeoutMs);
      this.waiters.push(w);
    });
  }

  _settle(w, v) {
    clearTimeout(w.timer);
    const i = this.waiters.indexOf(w);
    if (i >= 0) this.waiters.splice(i, 1);
    w.resolve(v);
  }

  _wake() {
    for (const w of [...this.waiters]) {
      if (w.gen !== this.core.gen) { this._settle(w, null); continue; }
      const s = this.core.slice(w.start);
      if (s.pcm) this._settle(w, s.pcm);
      else if (s.gone) this._settle(w, null);
    }
  }

  report() {
    const sp = this.core?.span();
    return {
      구간: sp ? `${sp.from.toFixed(2)}–${sp.to.toFixed(2)}s` : '—',
      대기: this.waiters.length,
      콜백평균_ms: this.calls ? +(this.costMs / this.calls).toFixed(3) : null,
    };
  }

  stop() {
    for (const [ev, fn] of this._listeners || []) this.video.removeEventListener(ev, fn);
    for (const w of [...this.waiters]) this._settle(w, null);
    if (this.node) {
      this.node.onaudioprocess = null;
      try { this.node.disconnect(); } catch { /* 이미 끊김 */ }
      try { this.player.audio?.src.disconnect(this.node); } catch { /* 재생기가 먼저 끊었다 */ }
    }
  }
};
