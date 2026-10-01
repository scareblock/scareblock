/**
 * 재생기 — 지연 재생과 개입.
 *
 * 인터페이스 계약의 **소비자**다. 탐지기가 누구든 아래 배열만 받는다.
 *
 *   [{ time, category, confidence, duration, source }]
 *
 * time은 **라이브 mediaTime**이다. 화면은 그보다 DELAY_SEC 뒤를 보여주므로,
 * 표시 시각이 그 구간에 들어올 때 블러가 걸린다. 탐지와 개입 사이의
 * 그 간격이 이 시스템의 존재 이유다.
 */
window.SB = window.SB || {};

SB.Player = class {
  constructor(video, cfg = SB.config) {
    this.video = video;
    this.cfg = cfg;
    this.triggers = [];
    this.running = false;

    // 계측 — M1 판정은 렌더끊김·디코더드롭 두 축으로 한다 (README M1 참고)
    this.stats = {
      missedFrames: 0,   // rVFC 콜백이 아예 보지 못한 원본 프레임 (사용자는 못 느낌)
      renderStalls: 0,   // 출력이 STALL_MS 넘게 멈춘 횟수
      renderFrames: 0,
      decoderDropsAtStart: 0,
      seeks: 0,
      resyncs: 0,
      startedAt: 0,
    };
    this._lastPresented = -1;
    this._lastCap = -1;
    this._cursor = 0;           // triggerAt()이 이미 지난 트리거를 다시 보지 않게 하는 자리
    this._lastRenderAt = 0;
    this._lastIncidentAt = 0;   // 마지막 끊김·디코더드롭 시각
    this._hudAt = 0;            // HUD·품질조회 스로틀
    this._drops = 0;            // 마지막으로 읽은 디코더드롭
    this._hudLine1 = '';
    this._hudLine2 = '';
    this._gainNow = 1;

    // 렌더 시간 계측 — eval 의 --render-s 를 실측으로 채우기 위한 것.
    // 블러 프레임과 일반 프레임을 나눠 재야 "블러가 얼마나 더 드는가"가 나온다.
    // 기본은 꺼둔다. 켜는 것은 `scareblock.measure(true)` 뿐이고, 데모·평상시
    // 경로에는 프레임당 비용이 없어야 한다.
    this._measuring = false;
    this._tBlurDraw = new SB.Samples();
    this._tPlainDraw = new SB.Samples();
    this._tBlurGap = new SB.Samples();
    this._tPlainGap = new SB.Samples();
    // 계측 구간 — 평균 프레임 주기를 세션 전체가 아니라 이 구간에서 낸다(#42 리뷰 🟡1).
    this._measureFrom = 0;      // 계측 시작 시점의 renderFrames
    this._measureAt = 0;        // 계측 시작 시각
    this._measureEnd = null;    // 계측을 끈 시점 { frames, at } — 켜져 있으면 null
    // 간격은 직전 프레임이 한 일의 비용이다. 직전 프레임의 블러 여부로 분류한다(#42 리뷰 🟡2).
    this._prevHit = null;       // null = 직전 프레임이 없거나 버퍼 채우는 중
  }

  /**
   * 계약 소비 — 탐지기가 부른다.
   *
   * 배열은 **time 오름차순을 유지한다.** triggerAt()의 상수 시간 조회가 그 순서에
   * 기댄다. 탐지기가 순서대로 주는 것이 보통이라 대개 뒤에 붙기만 한다.
   */
  addTriggers(list) {
    for (const t of list) {
      if (typeof t.time !== 'number' || typeof t.duration !== 'number') {
        console.warn('[scareblock] 계약에 맞지 않는 트리거', t);
        continue;
      }
      let i = this.triggers.length;
      while (i > 0 && this.triggers[i - 1].time > t.time) i--;
      this.triggers.splice(i, 0, t);
      if (i < this._cursor) this._cursor = i;   // 커서보다 앞에 끼면 되돌린다
    }
    const over = this.triggers.length - 500;
    if (over > 0) {
      this.triggers.splice(0, over);
      this._cursor = Math.max(0, this._cursor - over);
    }
  }

  /**
   * 표시 시각에 걸린 트리거. **매 렌더 프레임 불린다.**
   *
   * 선형 탐색이면 60fps × 500건 = 초당 3만 회 비교다. 프레임당 비용이 계측을
   * 실제로 망가뜨리는 것을 #21에서 확인했으므로 지난 트리거는 앞에서 잘라낸다.
   * 커서는 시크·탭 복귀에서 0으로 되돌린다 (그때 표시 시각이 뒤로 간다).
   */
  triggerAt(mediaTime) {
    const list = this.triggers;
    while (
      this._cursor < list.length &&
      list[this._cursor].time + list[this._cursor].duration < mediaTime
    ) {
      this._cursor++;
    }
    for (let i = this._cursor; i < list.length; i++) {
      const t = list[i];
      if (t.time > mediaTime) break;   // 오름차순이라 뒤는 볼 필요가 없다
      if (mediaTime <= t.time + t.duration) return t;
    }
    return null;
  }

  start() {
    const { video, cfg } = this;
    const slots = Math.ceil(cfg.DELAY_SEC * cfg.CAPTURE_FPS) + 10;
    this.ring = new SB.RingBuffer({ width: cfg.BUF_W, height: cfg.BUF_H, slots });

    const player = video.closest('#movie_player') || video.parentElement;
    const out = document.createElement('canvas');
    out.width = cfg.BUF_W;
    out.height = cfg.BUF_H;
    out.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;z-index:1;background:#000';
    player.appendChild(out);
    this.out = out;
    this.octx = out.getContext('2d', { alpha: false });

    this._prevOpacity = video.style.opacity;
    video.style.opacity = '0';

    this._setupAudio();
    this._bindResync();

    this.stats.decoderDropsAtStart = this._decoderDrops();
    this.stats.startedAt = performance.now();
    this.running = true;

    video.requestVideoFrameCallback(this._onFrame);
    requestAnimationFrame(this._render);
    SB.log(
      `지연 재생 시작 — ${cfg.DELAY_SEC}s / ${cfg.CAPTURE_FPS}fps / ` +
      `${cfg.BUF_W}x${cfg.BUF_H} / ${slots}슬롯 ≈ ${this.ring.estimatedMB}MB`
    );
  }

  /**
   * 오디오 그래프. 노드는 재사용하되 **연결은 매번 다시 세운다** —
   * stop()이 만든 직결이 남아 영상만 지연되던 버그가 있었다 (#5 리뷰 🔴1).
   */
  _setupAudio() {
    if (!SB._audio) {
      const ac = new AudioContext();
      SB._audio = {
        ac,
        el: null,
        src: null,
        delay: ac.createDelay(30),   // 0.25배속이면 DELAY_SEC/0.25 = 12초가 필요하다
        gain: ac.createGain(),
        analyser: Object.assign(ac.createAnalyser(), { fftSize: 1024 }),
      };
    }
    // MediaElementAudioSourceNode는 만들어질 때의 엘리먼트에 **영구히** 묶인다.
    // <video>가 교체되면 새 영상 소리는 그래프를 안 거치고 곧장 나가 지연이 사라진다
    // (#5 리뷰 🔴1과 같은 증상, 다른 원인). 어느 엘리먼트에 물렸는지 기억해 두고
    // 달라졌으면 소스만 다시 만든다 — AudioContext는 페이지당 개수 상한이 있어 재사용한다.
    if (SB._audio.el !== this.video) {
      try { SB._audio.src?.disconnect(); } catch { /* 아직 안 걸림 */ }
      try {
        SB._audio.src = SB._audio.ac.createMediaElementSource(this.video);
        SB._audio.el = this.video;
      } catch (e) {
        // 같은 엘리먼트에 두 번 물리면 InvalidStateError다. 페이지에서 다른
        // 스크립트(이전 PoC 스니펫 등)가 이미 잡았을 때 난다.
        throw new Error(`오디오 소스를 만들 수 없다 — ${e.name}: ${e.message}. ` +
          `이 페이지에서 이미 다른 스크립트가 <video>에 물렸을 수 있다. 새로고침해 본다.`);
      }
    }
    const { ac, src, delay, gain, analyser } = SB._audio;
    try {
      src.disconnect();
      delay.disconnect();
      gain.disconnect();
    } catch { /* 아직 안 걸림 */ }

    src.connect(analyser);          // 지연 전 — 탐지기가 미래를 먼저 듣는다
    src.connect(delay);
    delay.connect(gain);
    gain.connect(ac.destination);   // 사용자에게는 지연된 소리만
    gain.gain.value = 1;
    ac.resume();
    this.audio = SB._audio;
  }

  /**
   * 오디오 재동기화.
   *
   * 화면은 mediaTime 기준으로 DELAY_SEC 뒤를 보여주는데 DelayNode는 **벽시계**로
   * 지연한다. 정상 재생에서는 둘이 같지만 일시정지·배속·탭 숨김에서 어긋난다.
   * 셋 다 같은 뿌리라 한 곳에서 받는다.
   */
  _bindResync() {
    const v = this.video;

    // ① 일시정지 — 지연선에는 「화면이 앞으로 보여줄 3초」의 소리가 들어 있다.
    //    그대로 두면 멈춘 뒤에도 3초가 흘러나오고, 버리면 재개 후 3초가 무음이 된다.
    //    AudioContext를 통째로 멈추면 지연선 내용이 보존된다.
    this._onPause = () => { this.audio?.ac.suspend?.(); };
    this._onPlay = () => { this.audio?.ac.resume?.(); };

    // ② 배속 — 미디어 3초는 벽시계로 3/rate 초다.
    this._onRate = () => this._applyDelayForRate();

    // ③ 탭 숨김 — rAF·rVFC가 멈춰 링버퍼만 비는데 영상은 계속 간다.
    //    돌아오면 버퍼에 낡은 프레임만 남아 있어 과거 화면을 띄운다. 시크와 같게 다룬다.
    //    숨어 있는 동안 rVFC는 멈추지만 presentedFrames는 재생과 함께 계속 오른다.
    //    _lastPresented를 그대로 두면 돌아온 첫 콜백이 숨은 시간 전체를 적재누락으로
    //    적어 30초 전환에 1,800장이 한 번에 더해진다 — 누락_초당이 오염된다.
    this._onVisible = () => {
      if (document.visibilityState !== 'visible') return;
      this.ring.clear();
      this._lastCap = -1;
      this._lastPresented = -1;
      this._cursor = 0;
      // 같은 이유로 rAF도 멈춰 있었다. 복귀 첫 프레임의 간격(수 초)을 끊김으로
      // 세면 탭을 전환할 때마다 M1 연속 무드롭이 리셋된다.
      this._lastRenderAt = 0;
      this._prevHit = null;
      this.stats.resyncs++;
    };

    v.addEventListener('pause', this._onPause);
    v.addEventListener('play', this._onPlay);
    v.addEventListener('ratechange', this._onRate);
    document.addEventListener('visibilitychange', this._onVisible);
    this._applyDelayForRate();
  }

  _applyDelayForRate() {
    const a = this.audio;
    if (!a) return;
    const rate = this.video.playbackRate || 1;
    const param = a.delay.delayTime;

    // DelayNode에 maxDelayTime 속성은 없다. createDelay()에 넘기는 인자일 뿐이고,
    // 읽으려면 AudioParam.maxValue를 봐야 한다. 전에 a.delay.maxDelayTime을 썼다가
    // undefined - 0.1 = NaN 이 되어 램프가 던졌고, start() 안이라 화면이 까맣게 멈췄다.
    const ceiling = Number.isFinite(param.maxValue) ? param.maxValue - 0.1 : 10;
    const want = Math.min(this.cfg.DELAY_SEC / rate, ceiling);
    if (!Number.isFinite(want) || want < 0) {
      console.warn('[scareblock] 지연 값이 유효하지 않다', { rate, want, max: param.maxValue });
      return;
    }

    const now = a.ac.currentTime;
    if (a.ac.state === 'suspended') {
      // 정지 상태에서는 currentTime이 안 흘러 램프가 진행되지 않는다. 바로 넣는다.
      param.cancelScheduledValues(now);
      param.value = want;
      this._rateNow = rate;
      return;
    }
    if (Math.abs(param.value - want) < 0.01) { this._rateNow = rate; return; }

    // 지연선 **길이**를 바꾸면 읽기 지점이 건너뛴다. 램프로 늘리면 그 구간을
    // 되감느라 소리가 늘어지고(2배속 → 1배속에서 실측), 줄이면 건너뛴다.
    // 램프로는 숨길 수 없으므로 짧게 음소거하고 무음 구간에서 점프시킨다.
    const MUTE = 0.04;
    const g = a.gain.gain;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(0, now + MUTE);

    param.cancelScheduledValues(now);
    param.setValueAtTime(param.value, now);
    param.setValueAtTime(want, now + MUTE);      // 안 들리는 동안 갈아끼운다

    g.setValueAtTime(0, now + MUTE);
    g.linearRampToValueAtTime(this._gainNow ?? 1, now + MUTE * 2);

    this._rateNow = rate;
  }

  _unbindResync() {
    const v = this.video;
    v.removeEventListener('pause', this._onPause);
    v.removeEventListener('play', this._onPlay);
    v.removeEventListener('ratechange', this._onRate);
    document.removeEventListener('visibilitychange', this._onVisible);
  }

  _decoderDrops() {
    return this.video.getVideoPlaybackQuality?.().droppedVideoFrames ?? 0;
  }

  _onFrame = (_now, meta) => {
    if (!this.running) return;
    const t = meta.mediaTime;

    if (this._lastPresented >= 0 && meta.presentedFrames > this._lastPresented + 1) {
      this.stats.missedFrames += meta.presentedFrames - this._lastPresented - 1;
    }
    this._lastPresented = meta.presentedFrames;

    // 뒤로 시크하면 버퍼의 시간 순서가 깨져 이분 탐색이 성립하지 않는다. 비운다.
    if (t < this._lastCap) {
      this.ring.clear();
      this.stats.seeks++;
      this._lastCap = -1;
      this._lastPresented = -1;   // 시크 전후 델타는 누락이 아니다
      this._cursor = 0;           // 표시 시각이 뒤로 가므로 트리거 커서도 되돌린다
    }

    if (this._lastCap < 0 || t - this._lastCap >= 1 / this.cfg.CAPTURE_FPS - 0.002) {
      this.ring.push(this.video, t);
      this._lastCap = t;
    }
    this.video.requestVideoFrameCallback(this._onFrame);
  };

  _render = () => {
    if (!this.running) return;
    const { octx, cfg } = this;
    const now = performance.now();
    const gap = this._lastRenderAt ? now - this._lastRenderAt : 0;
    if (this._lastRenderAt && now - this._lastRenderAt > cfg.STALL_MS) {
      this.stats.renderStalls++;
      this._lastIncidentAt = now;
    }
    this._lastRenderAt = now;
    this.stats.renderFrames++;

    // getVideoPlaybackQuality()와 HUD 문자열 생성은 프레임마다 할 일이 아니다.
    // 4Hz로 낮춘다 — 프레임당 비용이 끊김의 원인이 될 수 있다.
    if (now - this._hudAt > 250) {
      this._hudAt = now;
      const d = this._decoderDrops() - this.stats.decoderDropsAtStart;
      if (d > this._drops) this._lastIncidentAt = now;
      this._drops = d;
      this._composeHud(d, now);
    }

    const slot = this.ring.findAtOrBefore(this.video.currentTime - cfg.DELAY_SEC);
    if (slot) {
      const hit = this.triggerAt(slot.t);
      const drawAt = this._measuring ? performance.now() : 0;
      octx.filter = hit ? `blur(${cfg.BLUR_PX}px)` : 'none';
      octx.drawImage(slot.canvas, 0, 0, cfg.BUF_W, cfg.BUF_H);
      octx.filter = 'none';
      if (this._measuring) {
        (hit ? this._tBlurDraw : this._tPlainDraw).push(performance.now() - drawAt);
        // gap은 직전 프레임 → 지금이라 직전 프레임의 비용이다. 이번 hit으로 나누면
        // 버스트 경계마다 한 프레임씩 반대 칸에 들어간다.
        if (gap > 0 && this._prevHit !== null) {
          (this._prevHit ? this._tBlurGap : this._tPlainGap).push(gap);
        }
        this._prevHit = !!hit;
      }
      const want = hit ? cfg.FADE_DB : 1;            // 음량 페이드다운
      if (want !== this._gainNow) {
        this._gainNow = want;
        this._rampGain(want);
      }
      if (hit) this._drawBadge(hit);
    } else {
      this._prevHit = null;
      octx.fillStyle = '#000';
      octx.fillRect(0, 0, cfg.BUF_W, cfg.BUF_H);
      octx.fillStyle = '#fff';
      octx.font = '20px system-ui, sans-serif';
      octx.fillText('안전 버퍼 채우는 중…', 28, 52);
    }
    this._drawHud();
    requestAnimationFrame(this._render);
  };

  /**
   * 음량을 램프로 바꾼다.
   *
   * 즉시 대입하면 진폭이 한 프레임에 1 → 0.25로 꺾여 파형에 불연속이 생기고
   * **딸깍 소리**가 난다. 페이드다운이 놀람 완화 기능 자체라 소리가 튀면
   * 기능이 기능을 해친다. delayTime에 쓴 것과 같은 방식이다.
   */
  _rampGain(want) {
    const a = this.audio;
    if (!a?.gain) return;
    const g = a.gain.gain;
    const now = a.ac.currentTime;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(want, now + 0.08);
  }

  _drawBadge(hit) {
    const { octx, cfg } = this;
    octx.fillStyle = 'rgba(0,0,0,.72)';
    octx.fillRect(cfg.BUF_W / 2 - 150, cfg.BUF_H / 2 - 26, 300, 52);
    octx.fillStyle = '#fff';
    octx.font = '22px system-ui, sans-serif';
    octx.textAlign = 'center';
    octx.fillText(`${hit.category} 가림`, cfg.BUF_W / 2, cfg.BUF_H / 2 + 8);
    octx.textAlign = 'left';
  }

  /** 4Hz로만 부른다. 문자열 생성과 품질 조회를 프레임에서 뺀다. */
  _composeHud(drops, now) {
    const { cfg, stats } = this;
    const sec = (now - stats.startedAt) / 1000;
    this._hudLine1 =
      `지연 ${cfg.DELAY_SEC}s · 버퍼 ${this.ring.filled}/${this.ring.slots} · ` +
      `트리거 ${this.triggers.length} · 시크 ${stats.seeks} · 재동기 ${stats.resyncs}` +
      (this._rateNow !== 1 ? ` · ${this._rateNow}x` : '');
    this._hudLine2 =
      `무드롭 ${this.cleanStreakSec(now).toFixed(0)}s · 끊김 ${stats.renderStalls} · ` +
      `드롭 ${drops} · 누락 ${stats.missedFrames} · ` +
      `${(stats.renderFrames / sec).toFixed(0)}fps`;
  }

  /** 마지막 끊김·드롭 이후 흐른 시간. M1의 「60초 이상 유지」가 이것이다. */
  cleanStreakSec(now = performance.now()) {
    return (now - Math.max(this.stats.startedAt, this._lastIncidentAt)) / 1000;
  }

  _drawHud() {
    const { octx, cfg } = this;
    octx.fillStyle = 'rgba(0,0,0,.6)';
    octx.fillRect(8, cfg.BUF_H - 52, 520, 44);
    octx.fillStyle = '#0f0';
    octx.font = '13px ui-monospace, monospace';
    octx.fillText(this._hudLine1, 14, cfg.BUF_H - 34);
    octx.fillText(this._hudLine2, 14, cfg.BUF_H - 16);
  }

  /**
   * 렌더 시간 — eval 의 `--render-s` 용.
   *
   * 블러 결정은 그 프레임을 그리는 rAF 콜백 안에서 일어나고, 픽셀은 다음
   * 합성 시점에 보인다. 그래서 **결정 → 화면**은 `draw 시간 + 한 프레임 주기`다.
   * 캔버스의 실제 제시 시각은 관측할 수 없으므로 이 값은 **근사**다.
   *
   * draw 시간은 CPU가 명령을 넣는 시간이라 GPU 완료를 포함하지 않는다.
   * 그래서 프레임 간격도 함께 본다 — 블러가 비싸면 간격이 늘어난다.
   */
  /** 계측 on/off. 켜면 표본을 비우고 새로 모은다. */
  measure(on = true) {
    this._measuring = !!on;
    if (on) {
      this._tBlurDraw.reset();
      this._tPlainDraw.reset();
      this._tBlurGap.reset();
      this._tPlainGap.reset();
      this._measureFrom = this.stats.renderFrames;
      this._measureAt = performance.now();
      this._measureEnd = null;
      this._prevHit = null;
    } else if (this._measureAt && !this._measureEnd) {
      this._measureEnd = { frames: this.stats.renderFrames, at: performance.now() };
    }
    SB.log(on ? '렌더 계측 켬 — 1분쯤 뒤 renderTiming()' : '렌더 계측 끔');
    return this._measuring;
  }

  renderTiming() {
    if (!this._measuring && !this._tBlurDraw.count) {
      return { 안내: '계측이 꺼져 있다. scareblock.measure(true) 로 켜고 1분쯤 뒤에 다시 부른다.' };
    }
    // 계측 구간만 쓴다. 세션 전체 평균이면 기동 직후 끊김 구간이 섞여 주기가
    // 길게 나오고 render_s가 부풀려진다.
    const end = this._measureEnd
      || { frames: this.stats.renderFrames, at: performance.now() };
    const sec = (end.at - this._measureAt) / 1000;
    const frames = end.frames - this._measureFrom;
    const frameMs = sec > 0 && frames > 0 ? 1000 * sec / frames : null;
    const blur = this._tBlurDraw.summary();
    const suggest = blur && frameMs ? (blur.p90 + frameMs) / 1000 : null;
    return {
      블러프레임_draw_ms: blur,
      일반프레임_draw_ms: this._tPlainDraw.summary(),
      블러프레임_간격_ms: this._tBlurGap.summary(),
      일반프레임_간격_ms: this._tPlainGap.summary(),
      평균프레임주기_ms: frameMs ? +frameMs.toFixed(2) : null,
      계측초: +sec.toFixed(1),
      'render_s(제안)': suggest ? +suggest.toFixed(4) : null,
      근거: 'render_s = p90(블러 draw) + 한 프레임 주기. 캔버스 제시 시각은 관측 불가라 근사값.',
    };
  }

  report() {
    const sec = (performance.now() - this.stats.startedAt) / 1000;
    return {
      해상도: `${this.cfg.BUF_W}x${this.cfg.BUF_H}@${this.cfg.CAPTURE_FPS}fps`,
      지연: this.cfg.DELAY_SEC,
      버퍼적재: `${this.ring.filled}/${this.ring.slots}`,
      메모리추정MB: this.ring.estimatedMB,
      경과초: +sec.toFixed(1),
      적재누락프레임: this.stats.missedFrames,
      렌더끊김: this.stats.renderStalls,
      디코더드롭: this._decoderDrops() - this.stats.decoderDropsAtStart,
      렌더fps: +(this.stats.renderFrames / sec).toFixed(1),
      시크: this.stats.seeks,
      재동기화: this.stats.resyncs,
      배속: this._rateNow ?? 1,
      트리거수: this.triggers.length,
      // 누적은 오래 돌수록 커진다. 비율로도 낸다.
      끊김_분당: +(this.stats.renderStalls / (sec / 60)).toFixed(2),
      누락_초당: +(this.stats.missedFrames / sec).toFixed(1),
      // M1 판정 — 「60초 이상 유지」는 누적 0이 아니라 연속 무드롭이다
      연속무드롭초: +this.cleanStreakSec().toFixed(1),
      M1통과: this.cleanStreakSec() >= 60,
    };
  }

  stop() {
    this.running = false;
    try { this._unbindResync(); } catch { /* _bindResync 전에 터졌을 수 있다 */ }
    this.audio?.ac.resume?.();   // 일시정지 상태로 두고 끄면 소리가 안 돌아온다
    this.out?.remove();
    this.video.style.opacity = this._prevOpacity ?? '';
    const a = SB._audio;
    if (a) {
      try {
        a.delay.disconnect();
        a.gain.disconnect();
        a.src.disconnect();
        a.src.connect(a.ac.destination);
      } catch { /* 이미 끊김 */ }
    }
    SB.log('원복 완료');
  }
};
