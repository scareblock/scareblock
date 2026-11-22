/**
 * 오디오를 YAMNet 입력(16 kHz 모노)으로 만드는 순수 계산 — 페이지 · 중계 · offscreen이 같이 쓴다.
 *
 *   SB.Resampler  — 임의 표본율 → 16 kHz. 스트리밍이고, 출력 표본 k는 **정확히** 입력 시각 k/16000초다
 *                   (필터 지연을 출력에 싣지 않는다 — 미래 입력이 반폭만큼 들어와야 내보낸다).
 *                   그래서 탐지 시각(미디어 시각)으로 패치를 잘라도 어긋나지 않는다
 *   SB.TapCore    — 16 kHz 표본을 최근 몇 초 들고 있고, 미디어 시각 구간으로 꺼낸다
 *   SB.pcm        — Float32Array ↔ base64. chrome.runtime 메시지는 JSON이라 형식 배열이 그대로 못 건넌다
 *
 * 브라우저 없이 `test/pcm.test.js`가 본다.
 */
window.SB = window.SB || {};

(() => {

  SB.YAMNET_SR = 16000;
  // 패치 0.96초가 실제로 덮는 오디오 — eval/yamnet.py WIN_S와 같다 (STFT 창 25 ms · 홉 10 ms)
  SB.YAMNET_WIN_S = 0.975;
  SB.YAMNET_WIN_N = Math.round(SB.YAMNET_WIN_S * SB.YAMNET_SR);   // 15600

  /**
   * 창 씌운 sinc 저역 통과 + 분수 위치 보간. 위상 표를 미리 만들어 콜백에서 sin을 부르지 않는다.
   * 48 kHz → 16 kHz면 위상이 늘 0이라 표 한 줄만 쓴다. 44.1 kHz는 위상이 바뀌어 가장 가까운 줄을 쓴다.
   */
  SB.Resampler = class {
    constructor(inRate, outRate = SB.YAMNET_SR, { zeros = 24, phases = 512 } = {}) {
      if (!(inRate > 0 && outRate > 0)) throw new Error(`표본율이 이상하다: ${inRate} → ${outRate}`);
      this.inRate = inRate;
      this.outRate = outRate;
      this.step = inRate / outRate;                  // 출력 한 칸 = 입력 몇 칸
      // 차단 주파수: 출력 나이퀴스트의 92% (입력 표본 단위, 사이클/표본)
      const fc = 0.5 * Math.min(1, outRate / inRate) * 0.92;
      this.half = Math.ceil(zeros / (2 * fc));       // 커널 반폭(입력 표본)
      this.phases = phases;
      const L = 2 * this.half + 1;
      this.table = new Float32Array(phases * L);
      for (let p = 0; p < phases; p++) {
        const frac = p / phases;                     // 출력 위치의 소수부
        let sum = 0;
        for (let j = 0; j < L; j++) {
          const d = j - this.half - frac;            // 입력 표본과 출력 위치의 거리
          const x = 2 * fc * d;
          const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
          const u = d / (this.half + 1);             // -1..1
          const w = Math.abs(u) >= 1 ? 0
            : 0.42 + 0.5 * Math.cos(Math.PI * u) + 0.08 * Math.cos(2 * Math.PI * u);  // 블랙먼
          const v = sinc * w;
          this.table[p * L + j] = v;
          sum += v;
        }
        // 직류 이득을 정확히 1로 — 위상마다 따로 맞춘다
        for (let j = 0; j < L; j++) this.table[p * L + j] /= sum;
      }
      this.L = L;
      this.reset();
    }

    reset() {
      this.buf = new Float32Array(Math.max(4096, this.L * 4));
      this.bufStart = -this.half;   // buf[0]의 절대 입력 위치. 처음 반폭은 0으로 채운다(시작 전은 무음)
      this.bufLen = this.half;
      this.k = 0;                   // 다음 출력 번호
    }

    /** 입력을 넣고 이번에 확정된 출력을 돌려준다. */
    push(x) {
      // 자리 확보 — 다음 출력에 필요 없는 앞부분은 버린다
      const keepFrom = Math.min(this.bufLen,
        Math.max(0, Math.floor(this.k * this.step) - this.half - this.bufStart));
      if (keepFrom > 0) {
        this.buf.copyWithin(0, keepFrom, this.bufLen);
        this.bufStart += keepFrom;
        this.bufLen -= keepFrom;
      }
      if (this.bufLen + x.length > this.buf.length) {
        const nb = new Float32Array(Math.max(this.buf.length * 2, this.bufLen + x.length + this.L));
        nb.set(this.buf.subarray(0, this.bufLen));
        this.buf = nb;
      }
      this.buf.set(x, this.bufLen);
      this.bufLen += x.length;

      const end = this.bufStart + this.bufLen;     // 들어온 입력의 끝(배타)
      const out = [];
      const { L, half, phases, table, step } = this;
      for (;;) {
        const pos = this.k * step;
        const i0 = Math.floor(pos);
        if (i0 + half >= end) break;               // 미래 반폭이 아직 없다
        let p = Math.round((pos - i0) * phases);
        let base = i0 - half;
        if (p === phases) { p = 0; base += 1; }
        const off = p * L;
        const b = base - this.bufStart;
        let acc = 0;
        for (let j = 0; j < L; j++) acc += table[off + j] * this.buf[b + j];
        out.push(acc);
        this.k++;
      }
      return Float32Array.from(out);
    }
  };

  /**
   * 16 kHz 표본의 고리. 미디어 시각과의 관계는 원점(anchor)과 배속 하나로 둔다 —
   * 규칙 탐지기(`detector-rule.js`)와 같은 방식이다. 원점이 깨지는 곳(시크 · 배속 변경 · 버퍼링)에서는
   * 호출자가 reset()한다.
   */
  SB.TapCore = class {
    constructor(inRate, keepS = 8) {
      this.inRate = inRate;
      this.cap = Math.round(keepS * SB.YAMNET_SR);
      this.ring = new Float32Array(this.cap);
      this.rs = new SB.Resampler(inRate);
      this.reset();
    }

    reset() {
      this.rs.reset();
      this.n = 0;              // reset() 이후 낸 16 kHz 표본 수
      this.anchor = null;      // 첫 입력 표본의 미디어 시각
      this.rate = 1;
      this.gen = (this.gen || 0) + 1;
    }

    /** 원점을 정한다. reset() 뒤 첫 push 전에 한 번. */
    setAnchor(media, rate = 1) {
      this.anchor = media;
      this.rate = rate || 1;
    }

    push(mono) {
      const y = this.rs.push(mono);
      for (let i = 0; i < y.length; i++) this.ring[(this.n + i) % this.cap] = y[i];
      this.n += y.length;
      return y.length;
    }

    /** 고리에 들어 있는 미디어 시각 범위 [from, to). 원점이 없으면 null. */
    span() {
      if (this.anchor === null) return null;
      const t = (k) => this.anchor + (k / SB.YAMNET_SR) * this.rate;
      return { from: t(Math.max(0, this.n - this.cap)), to: t(this.n) };
    }

    /**
     * 미디어 시각 `start`부터 오디오 n표본(기본 패치 하나 = 15600)을 꺼낸다.
     * 배속이어도 길이는 **오디오** 표본 수다 — YAMNet은 들리는 소리 0.975초를 본다.
     * @returns {{pcm: Float32Array} | {wait: true} | {gone: true}}
     */
    slice(start, n = SB.YAMNET_WIN_N) {
      if (this.anchor === null) return { wait: true };
      const k0 = Math.round(((start - this.anchor) / this.rate) * SB.YAMNET_SR);
      if (k0 < 0 || k0 < this.n - this.cap) return { gone: true };
      if (k0 + n > this.n) return { wait: true };
      const pcm = new Float32Array(n);
      for (let i = 0; i < n; i++) pcm[i] = this.ring[(k0 + i) % this.cap];
      return { pcm };
    }
  };

  SB.pcm = {
    /** Float32Array → base64 (리틀 엔디언 바이트 그대로). 손실 없다. */
    encode(f32) {
      const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
      let s = '';
      for (let i = 0; i < u8.length; i += 0x8000) {
        s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
      }
      return btoa(s);
    },

    decode(b64) {
      const s = atob(b64);
      const u8 = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i);
      if (u8.length % 4) throw new Error(`PCM 바이트 수가 4의 배수가 아니다: ${u8.length}`);
      return new Float32Array(u8.buffer, u8.byteOffset, u8.length / 4);
    },
  };
})();
