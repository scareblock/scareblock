/**
 * 페이지(MAIN world)에서 별도 문서 추론을 부르는 쪽.
 *
 * 왜 별도 문서인가 — 이 확장의 스크립트는 유튜브 페이지 안에서 돈다(MAIN world, 재생기가 <video>와
 * 오디오 그래프를 직접 잡아야 해서). 그래서 페이지 CSP를 그대로 받아 wasm을 올릴 수 없고,
 * `chrome.*` API도 없다. 추론은 확장 출처의 offscreen 문서에서 하고, 길은 이렇게 잇는다.
 *
 *   페이지(MAIN)  ─ window.postMessage ─▶  bridge.js(ISOLATED)  ─ chrome.runtime ─▶  offscreen.js
 *
 * 다른 페이지 스크립트도 같은 window 메시지를 본다. 오가는 것은 오디오 패치와 점수뿐이라 숨길 것은 없고,
 * 받는 쪽은 모양을 확인하고 모르는 것은 버린다.
 *
 * **실패하면 null이다.** 모델이 없거나 offscreen이 죽어도 탐지는 지금처럼 돈다 —
 * 부르는 쪽(말소리 기각 등)은 null이면 「검증 못 함 → 기각하지 않는다」로 처리한다(#40 클립 끝 처리와 같은 이유).
 */
window.SB = window.SB || {};

SB.INFER_TAG = 'scareblock-infer';

SB.InferClient = class {
  constructor(win = window, timeoutMs = 5000) {
    this.win = win;
    this.timeoutMs = timeoutMs;
    this.seq = 0;
    this.pending = new Map();
    this.stats = { 요청: 0, 성공: 0, 실패: 0, 시간초과: 0, 마지막오류: null, 왕복_ms: [] };
    this._onMsg = (e) => {
      if (e.source !== this.win) return;
      const m = e.data;
      if (!m || m.tag !== SB.INFER_TAG || m.dir !== 'res') return;
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      clearTimeout(p.timer);
      if (m.ok) {
        this.stats.성공++;
        this.stats.왕복_ms.push(Math.round(performance.now() - p.t0));
        if (this.stats.왕복_ms.length > 50) this.stats.왕복_ms.shift();
        p.resolve(m.result);
      } else {
        this.stats.실패++;
        this.stats.마지막오류 = m.error;
        p.resolve(null);
      }
    };
    this.win.addEventListener('message', this._onMsg);
  }

  /** @returns {Promise<any|null>} */
  call(op, payload = {}, timeoutMs = this.timeoutMs) {
    const id = ++this.seq;
    this.stats.요청++;
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        if (!this.pending.delete(id)) return;
        this.stats.시간초과++;
        this.stats.마지막오류 = `${op}: ${timeoutMs} ms 안에 응답 없음 — bridge.js가 없거나 offscreen이 안 떴다`;
        resolve(null);
      }, timeoutMs);
      this.pending.set(id, { resolve, timer, t0: performance.now() });
      this.win.postMessage({ tag: SB.INFER_TAG, dir: 'req', id, op, ...payload }, this.win.location.origin);
    });
  }

  /** offscreen 상태 — {ready, error, classes, ...} 또는 null. 첫 호출은 모델을 올리므로 넉넉히 기다린다 */
  status() { return this.call('status', {}, 20000); }

  /** 16 kHz 모노 0.975초 → 521개 클래스 점수(첫 패치) 또는 null */
  async yamnet(pcm) {
    if (!(pcm instanceof Float32Array) || pcm.length !== SB.YAMNET_WIN_N) {
      throw new Error(`YAMNet 입력은 16 kHz 표본 ${SB.YAMNET_WIN_N}개다 (받은 것: ${pcm?.length})`);
    }
    const r = await this.call('yamnet', { pcm });
    return r ? Float32Array.from(r.scores) : null;
  }

  report() {
    const rt = [...this.stats.왕복_ms].sort((a, b) => a - b);
    return {
      ...this.stats,
      왕복_ms: rt.length ? { 중앙값: rt[rt.length >> 1], 최대: rt[rt.length - 1], n: rt.length } : null,
      대기: this.pending.size,
    };
  }

  dispose() {
    this.win.removeEventListener('message', this._onMsg);
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.resolve(null); }
    this.pending.clear();
  }
};

/**
 * 탭 + 클라이언트를 묶은 것. 말소리 기각(#40)은 이것 하나만 부르면 된다:
 *   const s = await SB.yamnet.scoresAt(onset);   // null이면 검증 못 함
 *   s[0]  // Speech
 */
SB.Yamnet = class {
  constructor(tap, client) {
    this.tap = tap;
    this.client = client;
  }

  async scoresAt(mediaStart) {
    const pcm = await this.tap.patchAt(mediaStart);
    if (!pcm) return null;
    return this.client.yamnet(pcm);
  }
};
