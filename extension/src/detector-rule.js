/**
 * 규칙 기반 탐지기 — 02단계. `detector-fake.js` 자리에 들어온다.
 *
 * `eval/features.py` + `eval/detect.py`를 **한 프레임씩** 옮긴 것이다. E1이 잰
 * 것과 같은 탐지기가 브라우저에서 돌아야 E1의 숫자가 데모의 숫자가 된다.
 * 그래서 계산을 바꾸지 않는다 — 홉 10 ms · 창 25 ms · 직전 1초 중앙값 ·
 * 35초 반감 최댓값 · 되짚기 상한 1초. 동작점은 #35 사전 등록값 `surge_of_quiet = 20`.
 * 같은 wav에서 파이썬과 같은 탐지를 내는지는 `test/rule-parity.py`가 본다.
 *
 * 둘로 나눈다.
 *   SB.RuleCore     — 순수 계산. 샘플을 받아 탐지를 낸다. 브라우저 없이 테스트한다
 *   SB.RuleDetector — 오디오 그래프에 물려 탐지를 계약 배열로 재생기에 넘긴다
 */
window.SB = window.SB || {};

/** `eval/detect.py` Params와 같은 이름·같은 뜻. surge_of_quiet만 동작점으로 바꾼다. */
SB.RULE_PARAMS = {
  quiet_of_peak: 0.15,
  surge_of_quiet: 20.0,    // #35 사전 등록 동작점 (27초에 한 번)
  surge_of_peak: 0.30,
  drms_min: 0.0,
  cooldown_s: 2.0,
  silence_floor: 1e-4,
  backtrack_max_s: 1.0,
};

SB.RuleCore = class {
  constructor(sampleRate, params = SB.RULE_PARAMS, hopS = 0.010, winS = 0.025) {
    this.p = { ...SB.RULE_PARAMS, ...params };
    this.sr = sampleRate;
    this.hopS = hopS;
    // features.extract()와 같은 반올림
    this.hop = Math.max(1, Math.round(sampleRate * hopS));
    this.win = Math.max(this.hop, Math.round(sampleRate * winS));
    this.quietW = Math.max(1, Math.round(1.0 / hopS));
    this.decay = 0.5 ** (hopS / 35.0);
    this.cooldownFrames = Math.round(this.p.cooldown_s / hopS);
    this.backFrames = Math.max(1, Math.round(this.p.backtrack_max_s / hopS));
    // 되짚기가 1초 + 중앙값 창이 1초 — RMS·정적·최댓값을 그만큼 들고 있는다
    this.keep = this.quietW + this.backFrames + 2;
    this.reset();
  }

  /** 시크 뒤에는 이전 위치의 정적·최댓값이 의미가 없다. 클립 처음처럼 다시 시작한다. */
  reset() {
    this.buf = new Float32Array(this.win + this.hop * 64);
    this.bufLen = 0;
    this.bufStart = 0;          // buf[0]의 절대 샘플 위치
    this.i = 0;                 // 다음에 계산할 프레임 번호
    this.rms = [];              // 최근 keep개 — 인덱스 = 프레임 번호 - this.base
    this.quiet = [];
    this.peakArr = [];
    this.base = 0;
    this.peak = 0;
    this.prevRms = null;
    this.blockedUntil = -1;
  }

  /**
   * 모노 샘플을 넣는다. 이번에 확정된 탐지를 돌려준다.
   * 시각은 전부 **reset() 이후 첫 샘플 기준 초**다. 미디어 시각으로 바꾸는 것은 호출자 몫이다.
   * @returns {{onset:number, fire:number, score:number, onsetCapped:boolean}[]}
   */
  push(samples) {
    const out = [];
    let k = 0;
    while (k < samples.length) {
      const room = this.buf.length - this.bufLen;
      const n = Math.min(room, samples.length - k);
      this.buf.set(samples.subarray(k, k + n), this.bufLen);
      this.bufLen += n;
      k += n;
      this._drain(out);
    }
    return out;
  }

  _drain(out) {
    // 프레임 i의 창 = [i*hop, i*hop + win)
    for (;;) {
      const s0 = this.i * this.hop - this.bufStart;
      if (s0 + this.win > this.bufLen) break;
      let acc = 0;
      for (let j = s0; j < s0 + this.win; j++) acc += this.buf[j] * this.buf[j];
      this._frame(Math.sqrt(acc / this.win), out);
      this.i++;
    }
    // 다음 프레임 창 앞은 버린다
    const drop = this.i * this.hop - this.bufStart;
    if (drop > 0) {
      this.buf.copyWithin(0, drop, this.bufLen);
      this.bufLen -= drop;
      this.bufStart += drop;
    }
  }

  _at(arr, i) { return arr[i - this.base]; }

  _frame(cur, out) {
    const i = this.i;
    const p = this.p;

    // 직전 1초 중앙값 — 현재 프레임은 넣지 않는다 (features.py와 같다)
    const lo = Math.max(0, i - this.quietW);
    let quiet;
    if (i > lo) {
      const w = this.rms.slice(lo - this.base, i - this.base).sort((a, b) => a - b);
      const m = w.length >> 1;
      quiet = w.length % 2 ? w[m] : (w[m - 1] + w[m]) / 2;
    } else {
      quiet = cur;
    }
    this.peak = Math.max(cur, this.peak * this.decay);
    const drms = (this.prevRms === null ? 0 : cur - this.prevRms) / this.hopS;
    this.prevRms = cur;

    this.rms.push(cur);
    this.quiet.push(quiet);
    this.peakArr.push(this.peak);
    if (this.rms.length > this.keep) {
      const d = this.rms.length - this.keep;
      this.rms.splice(0, d);
      this.quiet.splice(0, d);
      this.peakArr.splice(0, d);
      this.base += d;
    }

    // detect.run()의 조건 그대로
    const peak = this.peak;
    if (i < this.blockedUntil || peak < p.silence_floor) return;
    if (!(quiet < peak * p.quiet_of_peak)) return;
    if (!(cur > Math.max(quiet * p.surge_of_quiet, peak * p.surge_of_peak))) return;
    if (p.drms_min > 0 && drms < p.drms_min) return;

    // _backtrack_onset()
    const floor = Math.max(quiet * 1.5, peak * 0.02);
    const limit = Math.max(0, i - this.backFrames);
    let j = i;
    while (j > limit && this._at(this.rms, j) > floor) j--;
    const capped = j === limit && this._at(this.rms, j) > floor;

    out.push({
      onset: j * this.hopS,
      fire: i * this.hopS + this.win / this.sr,   // t_ready — 창의 끝
      score: cur / (peak + 1e-9),
      onsetCapped: capped,
    });
    this.blockedUntil = i + this.cooldownFrames;
  }
};

/**
 * 오디오 그래프에 물리는 쪽.
 *
 * **ScriptProcessorNode를 쓴다.** AudioWorklet이 정석이지만 모듈을 URL로 불러와야 하고,
 * 이 스크립트는 유튜브 페이지(MAIN world)에서 돌아 페이지의 CSP·Trusted Types를 그대로
 * 받는다. 이 노드는 출력을 내지 않고 듣기만 하며, 2048 샘플(48 kHz에서 43 ms)마다 한 번
 * 메인 스레드에서 돈다 — 탐지 지연이 그만큼 늘지만 3초 지연 안에서는 무시할 크기다.
 *
 * 지연 **전** 소스에 물린다. 재생기의 analyser와 같은 자리다 — 탐지기가 미래를 먼저 듣는다.
 */
SB.RuleDetector = class {
  constructor(video, player, params = SB.RULE_PARAMS) {
    this.video = video;
    this.player = player;
    this.params = params;
    this.n = 0;
  }

  start() {
    const { ac, src } = this.player.audio;
    this.core = new SB.RuleCore(ac.sampleRate, this.params);
    this.node = ac.createScriptProcessor(2048, 2, 1);
    this.mono = new Float32Array(2048);
    this._anchor = null;   // reset() 이후 첫 콜백의 미디어 시각 — 탐지 시각의 원점

    this.node.onaudioprocess = (e) => {
      if (this.video.paused || this.video.seeking) return;
      const inp = e.inputBuffer;
      const a = inp.getChannelData(0);
      const b = inp.numberOfChannels > 1 ? inp.getChannelData(1) : a;
      for (let k = 0; k < a.length; k++) this.mono[k] = (a[k] + b[k]) / 2;

      // 이 버퍼의 첫 샘플이 들어온 미디어 시각. 원점은 한 번만 잡고 이후는 샘플 수로
      // 센다 — 콜백마다 currentTime을 다시 읽으면 그 흔들림이 탐지 시각에 그대로 실린다.
      const rate = this.video.playbackRate || 1;
      if (this._anchor === null) {
        // 배속이면 이 버퍼가 덮는 미디어 시간은 inp.duration * rate다 (#46 리뷰 🟡1)
        this._anchor = this.video.currentTime - inp.duration * rate;
      }
      for (const d of this.core.push(this.mono.subarray(0, a.length))) this._emit(d, rate);
    };

    // 원점과 샘플 수로 미디어 시각을 세므로, 둘의 관계가 깨지는 곳에서 다시 시작한다.
    //   seeked     — 이전 위치의 정적·최댓값이 남으면 새 위치의 첫 소리가 「정적 뒤 급등」으로 걸린다
    //   ratechange — 원점 이후 배속이 하나라는 가정이 깨진다
    //   playing    — **버퍼링(waiting)을 거쳤을 때만.** 그동안 미디어는 멈췄는데 무음 샘플은 계속 세어진다.
    //                사용자 일시정지 뒤의 playing은 리셋하지 않는다 — 멈춘 동안은 콜백이 샘플을 넣지 않아
    //                원점과 샘플 수의 관계가 그대로이고, 리셋하면 직전 1초 정적 이력만 잃는다 (#46 리뷰 🟡2)
    this._onReset = () => { this.core.reset(); this._anchor = null; this._stalled = false; };
    this._onWaiting = () => { this._stalled = true; };
    this._onPlaying = () => { if (this._stalled) this._onReset(); };
    this._listeners = [
      ['seeked', this._onReset], ['ratechange', this._onReset],
      ['waiting', this._onWaiting], ['playing', this._onPlaying],
    ];
    for (const [ev, fn] of this._listeners) this.video.addEventListener(ev, fn);

    src.connect(this.node);
    // 목적지에 이어야 Chrome이 콜백을 부른다. 출력 버퍼를 건드리지 않으므로 무음이다.
    this.node.connect(ac.destination);
    SB.log(`규칙 탐지기 시작 — surge_of_quiet=${this.core.p.surge_of_quiet} · ${ac.sampleRate} Hz`);
  }

  _emit(d, rate) {
    // 배속이면 오디오 1초가 미디어 rate초다
    const onset = this._anchor + d.onset * rate;
    const fire = this._anchor + d.fire * rate;
    this.n++;
    this.player.addTriggers([{
      time: onset,
      category: 'jumpscare',
      confidence: Math.min(1, d.score),
      duration: SB.config.TRIGGER_TAIL_S,
      source: 'rule',
    }]);
    SB.log(
      `규칙 탐지 #${this.n} onset=${onset.toFixed(2)}s fire=${fire.toFixed(2)}s ` +
      `(지금 ${this.video.currentTime.toFixed(2)}s) score=${d.score.toFixed(2)}` +
      (d.onsetCapped ? ' · 되짚기 상한' : '')
    );
  }

  /**
   * 데모용 — 동작점을 실행 중에 바꾼다. **E1 보고값은 20 그대로다(#35 사전 등록).**
   * 낮추면 더 잡지만 일반 장면에서도 자주 울린다. 그 트레이드오프를 보여주려고 연다.
   * 앞으로 들어오는 프레임부터 적용되고, 정적·최댓값 이력은 그대로 이어 쓴다.
   */
  setSurge(n) {
    if (!(Number.isFinite(n) && n > 1)) {
      console.warn('[scareblock] surge_of_quiet는 1보다 큰 숫자여야 한다', n);
      return this.core?.p.surge_of_quiet;
    }
    if (this.core) this.core.p.surge_of_quiet = n;
    else this.params = { ...this.params, surge_of_quiet: n };
    SB.log(`surge_of_quiet = ${n}` + (n === 20 ? ' (E1 동작점)' : ' — E1 동작점(20)과 다르다'));
    return n;
  }

  stop() {
    for (const [ev, fn] of this._listeners || []) this.video.removeEventListener(ev, fn);
    if (this.node) {
      this.node.onaudioprocess = null;
      try { this.node.disconnect(); } catch { /* 이미 끊김 */ }
      try { this.player.audio?.src.disconnect(this.node); } catch { /* 재생기가 먼저 끊었다 */ }
    }
  }
};
