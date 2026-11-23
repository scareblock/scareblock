/**
 * E5 사용자 실험 진행 도우미 — 클립 시작 · 끝을 두 조건에서 같게 맞춘다 (#35 E5 등록).
 *
 * **확장에 기대지 않는다.** 원본 조건은 확장을 끈 프로필에서 보므로, 이 파일은 확장 없이도 돈다.
 * 확장이 있으면(`window.scareblock`) 필터 조건으로 보고, 화면이 3초 늦게 나오는 것을 계산에 넣는다.
 *
 * 쓰는 법 — DevTools(⌘⌥J) → Sources → Snippets에 한 번 저장해 두고, 클립마다 ⌘P → `!e5`로 실행한다.
 *   e5.prepare('c019')            // 시작점에 맞춰 멈춰 두고 검은 화면에 「준비됨」을 띄운다
 *   (참가자가 앉으면) Space         // 검은 화면이 걷히고 재생. 300초가 「보이면」 멈추고 「끝」
 *   e5.report()                   // 이번 세션 기록 — 실제 시작 · 끝 시각, 중간에 끼어든 일시정지 · 시크
 *   Esc                           // 진행 중 참가자가 그만두고 싶을 때 — 기록은 남고 그 세션은 무효
 *   e5.abort()                    // 진행 취소, 화면 원래대로
 *
 * 필터 조건에서 「안전 버퍼 채우는 중…」을 참가자가 보지 않게, 시작점보다 1초 앞에서 미리 틀어 (지연 + 1)초 동안
 * 버퍼를 채운 뒤 **화면에 시작 프레임이 보이는 순간** 멈춘다. 멈춰도 재생기가 지연선 소리 3초를 그대로 들고 있다
 * (player.js 「오디오 재동기화」 ①). 이 준비 동안 4초쯤 소리가 나므로 참가자는 헤드폰을 쓰기 전이어야 한다.
 *
 * 시작 전에 확인할 것: 안정적인 볼륨 끔(#35 · #52 — 켜면 규칙 탐지가 절반으로 준다), 광고 없는 계정, 음량.
 */
(() => {
  const g = typeof window !== 'undefined' ? window : globalThis;

  // clips-e1.csv의 video_id · clip_offset. 저장소에 없는 파일이라 실험 전에 채운다 (#35 E5 클립)
  const CLIPS = {
    c019: { video: null, start: null },
    c004: { video: null, start: null },
  };
  const DUR = 300;      // 클립 길이(초) — E1 클립과 같다
  const LEAD = 1.0;     // 필터 조건: 시작점보다 LEAD초 앞에서 튼다 — 화면이 시작점을 보일 때 버퍼에 그 앞 LEAD초도 있다

  class E5Session {
    /**
     * @param {object} env  { video, host, doc, win, delay, videoId, log }
     *   delay — 화면이 mediaTime보다 몇 초 늦게 보이는가 (원본 0, 필터 SB.config.DELAY_SEC)
     */
    constructor(env) {
      Object.assign(this, env);
      this.state = 'idle';
      this.events = [];
    }

    get condition() { return this.delay > 0 ? '필터' : '원본'; }

    /** 화면에 보이는 미디어 시각 */
    shown(mediaTime) { return mediaTime - this.delay; }

    prepare(clip) {
      if (this.state === 'running') throw new Error('진행 중이다 — 먼저 e5.abort()');
      const c = typeof clip === 'string' ? CLIPS[clip] : clip;
      if (!c || typeof c.start !== 'number') {
        throw new Error(`${clip}의 시작점이 없다 — e5-session.js의 CLIPS에 clips-e1.csv 값을 넣거나 e5.prepare({start, video})`);
      }
      if (c.video && this.videoId && c.video !== this.videoId) {
        throw new Error(`이 페이지 영상(${this.videoId})이 ${clip}(${c.video})가 아니다`);
      }
      this.clip = { id: typeof clip === 'string' ? clip : 'custom', start: c.start, dur: c.dur ?? DUR };
      this.events = [];
      this._note('prepare', { 조건: this.condition, ...this.clip });
      this._overlay('준비 중…');
      this._hideChrome(true);
      this.state = 'priming';

      const v = this.video;
      const from = this.clip.start - (this.delay > 0 ? LEAD : 0);
      v.pause();
      v.currentTime = Math.max(0, from);
      if (this.delay > 0) {
        // 버퍼를 채운다: 화면에 시작 프레임이 보이는 순간 멈춘다
        const onSeeked = () => {
          v.removeEventListener('seeked', onSeeked);
          this._watch((t) => this.shown(t) >= this.clip.start, (t) => this._primed(t));
          v.play();
        };
        v.addEventListener('seeked', onSeeked);
      } else {
        const onSeeked = () => { v.removeEventListener('seeked', onSeeked); this._primed(v.currentTime); };
        v.addEventListener('seeked', onSeeked);
      }
      return `${this.condition} 조건 · ${this.clip.id} ${this.clip.start}s부터 ${this.clip.dur}s — 준비 중`;
    }

    _primed(t) {
      this.video.pause();
      this.state = 'ready';
      this.tReady = t;
      this._note('ready', { 보이는시각: +this.shown(t).toFixed(3) });
      this._overlay('준비됨 — Space 키로 시작');
      this._keys(true);
      this.log(`[e5] 준비됨 — 보이는 시각 ${this.shown(t).toFixed(2)}s. 참가자가 준비되면 Space`);
    }

    start() {
      if (this.state !== 'ready') return `지금은 ${this.state} — prepare 뒤 준비됨에서만 시작한다`;
      this.state = 'running';
      this._overlay('', { clear: true });   // 화면은 비우되 클릭은 계속 막는다 — 영상 클릭이 재생 · 정지를 바꾼다
      this._guard(true);
      this.t0 = this.now();
      this._note('start', { 보이는시각: +this.shown(this.tReady ?? this.video.currentTime).toFixed(3) });
      const end = this.clip.start + this.clip.dur;
      this._watch((t) => this.shown(t) >= end, (t) => this._end(t));
      this.video.play();
      return 'started';
    }

    _end(t) {
      this._guard(false);          // 내가 멈추는 것은 끼어든 일이 아니다 — 감시를 먼저 끈다
      this.state = 'done';
      this._keys(false);
      this.video.pause();
      const wall = (this.now() - this.t0) / 1000;
      // 멈춘 뒤 currentTime이 아니라 판정한 프레임의 시각을 남긴다 — 둘이 수 ms 다르다
      this._note('end', { 보이는시각: +this.shown(t).toFixed(3), 걸린초: +wall.toFixed(2) });
      this._overlay('끝');
      this.log(`[e5] 끝 — ${this.clip.id} ${this.condition}, ${wall.toFixed(1)}초. e5.report()로 기록 확인`);
    }

    abort() {
      this._stopWatch();
      this._keys(false);
      this._guard(false);
      this._overlay(null);
      this._hideChrome(false);
      this.video.pause();
      this._note('abort');
      this.state = 'idle';
    }

    report() {
      const by = (k) => this.events.find((e) => e.k === k);
      const s = by('start'), e = by('end');
      const extra = this.events.filter((x) => x.k === 'interrupt');
      return {
        조건: this.condition,
        클립: this.clip?.id,
        상태: this.state,
        시작_보이는시각: s?.보이는시각,
        끝_보이는시각: e?.보이는시각,
        걸린초: e?.걸린초,
        끼어든일: extra.map((x) => `${x.what} @${x.보이는시각}s`),
        유효: !!(s && e && extra.length === 0),
      };
    }

    // ── 내부 ──────────────────────────────────────────────
    _note(k, o = {}) { this.events.push({ k, ...o }); }

    /** 프레임마다 mediaTime을 보고 조건이 맞으면 한 번 부른다 */
    _watch(cond, fire) {
      this._stopWatch();
      const id = {};
      this._watchId = id;
      const tick = (_now, meta) => {
        if (this._watchId !== id) return;
        if (cond(meta.mediaTime)) { this._watchId = null; fire(meta.mediaTime); return; }
        this.video.requestVideoFrameCallback(tick);
      };
      this.video.requestVideoFrameCallback(tick);
    }

    _stopWatch() { this._watchId = null; }

    /** 재생 중에 누가 멈추거나 옮기면 기록한다 — 그 세션은 유효하지 않다 */
    _guard(on) {
      const v = this.video;
      if (on) {
        this._onPause = () => { if (this.state === 'running') this._note('interrupt', { what: '일시정지', 보이는시각: +this.shown(v.currentTime).toFixed(2) }); };
        this._onSeek = () => { if (this.state === 'running') this._note('interrupt', { what: '시크', 보이는시각: +this.shown(v.currentTime).toFixed(2) }); };
        this._onRate = () => { if (this.state === 'running') this._note('interrupt', { what: `배속 ${v.playbackRate}`, 보이는시각: +this.shown(v.currentTime).toFixed(2) }); };
        v.addEventListener('pause', this._onPause);
        v.addEventListener('seeking', this._onSeek);
        v.addEventListener('ratechange', this._onRate);
      } else if (this._onPause) {
        v.removeEventListener('pause', this._onPause);
        v.removeEventListener('seeking', this._onSeek);
        v.removeEventListener('ratechange', this._onRate);
        this._onPause = null;
      }
    }

    /**
     * 준비됨 · 진행 중에는 유튜브 단축키가 영상을 건드리지 못하게 가로챈다.
     * **keydown만 막으면 안 된다** — 유튜브는 Space를 keyup에서도 받아, 시작한 직후 다시 멈췄다(11.23 실기).
     *   준비됨: Space → 시작
     *   진행 중: Esc → 참가자 중단(동의서: 언제든 멈출 수 있다). 나머지 재생 단축키는 무시
     *   f(전체 화면)는 막지 않는다
     */
    _keys(on) {
      const BLOCK = new Set([' ', 'k', 'K', 'j', 'J', 'l', 'L', 'm', 'M', ',', '.', '<', '>', 'Home', 'End',
        'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
      if (on && !this._onKey) {
        this._onKey = (e) => {
          if (this.state !== 'ready' && this.state !== 'running') return;
          const space = e.code === 'Space' || e.key === ' ';
          if (e.type === 'keydown' && space && this.state === 'ready') {
            e.preventDefault(); e.stopImmediatePropagation();
            this.start();
            return;
          }
          if (e.type === 'keydown' && e.key === 'Escape' && this.state === 'running') {
            e.preventDefault(); e.stopImmediatePropagation();
            this.stopByParticipant();
            return;
          }
          if (space || BLOCK.has(e.key)) { e.preventDefault(); e.stopImmediatePropagation(); }
        };
        for (const t of ['keydown', 'keyup', 'keypress']) this.win.addEventListener(t, this._onKey, true);
      } else if (!on && this._onKey) {
        for (const t of ['keydown', 'keyup', 'keypress']) this.win.removeEventListener(t, this._onKey, true);
        this._onKey = null;
      }
    }

    /** 참가자가 그만두고 싶다고 할 때 — Esc. 기록은 남기고 그 세션은 유효하지 않다 */
    stopByParticipant() {
      if (this.state !== 'running') return;
      this._note('interrupt', { what: '참가자 중단(Esc)', 보이는시각: +this.shown(this.video.currentTime).toFixed(2) });
      this._stopWatch();
      this._guard(false);
      this.state = 'stopped';
      this.video.pause();
      this._overlay('중단했습니다');
      this.log('[e5] 참가자 중단 — 이 세션은 분석에서 빠진다(#35 E5 「중간에 멈춘 사람」)');
    }

    /** 플레이어 안에 덮는다 — 유튜브 전체 화면(f)에서도 보이도록 #movie_player 안에 둔다 */
    _overlay(text, { clear = false } = {}) {
      if (!this._ov) {
        const d = this.doc.createElement('div');
        d.style.cssText = 'position:absolute;inset:0;z-index:10000;color:#fff;' +
          'display:flex;align-items:center;justify-content:center;font:28px system-ui,sans-serif;cursor:none';
        // 플레이어를 클릭하면 유튜브가 재생 · 정지를 바꾼다 — 덮개가 받아서 버린다
        const swallow = (e) => { e.preventDefault(); e.stopPropagation(); };
        for (const t of ['click', 'dblclick', 'mousedown', 'mouseup', 'pointerdown', 'pointerup', 'contextmenu']) {
          d.addEventListener(t, swallow);
        }
        this._ov = d;
      }
      if (text == null) { this._ov.remove?.(); return; }
      this._ov.style.background = clear ? 'transparent' : '#000';
      this._ov.textContent = text;
      if (!this._ov.parentNode) this.host.appendChild(this._ov);
    }

    /** 진행 막대 · 제목 줄이 남은 시간을 알려주지 않게 숨긴다 */
    _hideChrome(on) {
      if (on && !this._style) {
        const s = this.doc.createElement('style');
        s.textContent = '.ytp-chrome-top,.ytp-chrome-bottom,.ytp-gradient-top,.ytp-gradient-bottom,' +
          '.ytp-ce-element,.ytp-pause-overlay{display:none!important}#movie_player{cursor:none}';
        this.doc.head.appendChild(s);
        this._style = s;
      } else if (!on && this._style) {
        this._style.remove?.();
        this._style = null;
      }
    }
  }

  g.E5Session = E5Session;
  g.E5_CLIPS = CLIPS;

  // 페이지에서 실행됐을 때만 전역 도우미를 만든다 (테스트는 E5Session을 직접 쓴다)
  if (typeof document !== 'undefined' && document.querySelector) {
    const video = document.querySelector('video.html5-main-video') || document.querySelector('video');
    if (!video) { console.warn('[e5] <video>가 없다 — 유튜브 영상 페이지에서 실행한다'); return; }
    g.e5?.abort?.();
    const filtered = !!g.scareblock?.player;
    const s = new E5Session({
      video,
      host: video.closest('#movie_player') || video.parentElement,
      doc: document,
      win: g,
      delay: filtered ? (g.SB?.config?.DELAY_SEC ?? 3) : 0,
      videoId: new URLSearchParams(location.search).get('v'),
      now: () => performance.now(),
      log: (...a) => console.log(...a),
    });
    g.e5 = {
      prepare: (c) => s.prepare(c),
      start: () => s.start(),
      report: () => s.report(),
      abort: () => s.abort(),
      stop: () => s.stopByParticipant(),
      session: s,
    };
    console.log(`[e5] ${s.condition} 조건 (${filtered ? '확장 있음 — 화면 3초 지연' : '확장 없음'}). ` +
      '안정적인 볼륨 끔 · 광고 없음 · 음량 확인 뒤 e5.prepare(\'c019\')');
  }
})();
