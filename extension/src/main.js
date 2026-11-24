/** 조립. 유튜브가 SPA라 영상이 바뀌어도 따라붙어야 한다. */
window.SB = window.SB || {};

(() => {
  let player = null;
  let detector = null;
  let visual = null;
  let tap = null;
  let clip = null;
  let clipFrame = null;   // 페이지에 하나 — 영상이 바뀌어도 프레임은 그대로 쓴다
  // 클라이언트는 페이지에 하나 — 영상이 바뀌어도 window 메시지 리스너를 다시 걸지 않는다
  const infer = (SB.config.YAMNET || SB.config.CLIP) ? new SB.InferClient() : null;

  /**
   * offscreen이 준비됐으면 오디오 탭을 붙이고 SB.yamnet을 연다. 아니면 아무것도 붙이지 않는다 —
   * 탐지는 지금처럼 돈다. 확인은 비동기라 attach()를 늦추지 않는다.
   */
  async function attachYamnet(video, p) {
    const st = await infer.status();
    if (p !== player || !p?.running) return;          // 기다리는 사이 영상이 바뀌었다
    if (!st?.ready) {
      SB.log(`YAMNet 꺼짐 — ${st?.error || infer.stats.마지막오류 || '응답 없음'}`);
      return;
    }
    try {
      tap = new SB.AudioTap(video, p);
      tap.start();
      SB.yamnet = new SB.Yamnet(tap, infer);
      SB.log(`YAMNet 준비 — 모델 로드 ${st.loadMs} ms`);
    } catch (e) {
      console.error('[scareblock] 오디오 탭 시작 실패 — YAMNet 없이 돈다', e);
      tap = null;
      SB.yamnet = null;
    }
  }

  /** offscreen에 CLIP이 올라왔으면 샘플러를 붙인다. 아니면 아무것도 안 붙인다 — 오디오 탐지는 그대로 */
  async function attachClip(video, p) {
    const st = await infer.clipStatus();
    if (p !== player || !p?.running) return;
    if (!st?.ready) {
      SB.log(`CLIP 꺼짐 — ${st?.error || infer.stats.마지막오류 || '응답 없음'}`);
      return;
    }
    // 자르기는 확장 프레임에서 한다(clip-frame.js) — 프레임이 안 붙으면 CLIP 없이 돈다
    clipFrame ||= new SB.ClipFrameClient();
    if (!(await clipFrame.ready())) {
      SB.log(`CLIP 꺼짐 — ${clipFrame.lastError}`);
      return;
    }
    if (p !== player || !p?.running) return;
    try {
      clip = new SB.ClipSampler(video, p, clipFrame);
      clip.start();
      SB.log(`CLIP 준비 — 모델 로드 ${st.loadMs} ms · 피 ${st.probe === 'blood' ? '보강 모델' : 'zero-shot'}`);
    } catch (e) {
      console.error('[scareblock] CLIP 시작 실패 — 없이 돈다', e);
      clip = null;
    }
  }

  function stopYamnet() {
    tap?.stop();
    tap = null;
    SB.yamnet = null;
  }

  // ── 사용자 설정 ── 중계가 chrome.storage에서 읽어 넘겨 준다(settings.js). 오기 전에는 기본값
  let settings = SB.normalizeSettings(null);
  let gotSettings = null;
  const settingsReady = new Promise((res) => { gotSettings = res; setTimeout(res, 500); });   // 중계가 없어도 0.5초 뒤 기본값으로 시작
  window.addEventListener('message', (e) => {
    if (e.source !== window || e.data?.tag !== SB.SETTINGS_TAG || e.data.dir !== 'push') return;
    const prev = settings;
    settings = SB.normalizeSettings(e.data.settings);
    gotSettings();
    onSettings(prev);
  });
  window.postMessage({ tag: SB.SETTINGS_TAG, dir: 'get' }, location.origin);

  function onSettings(prev) {
    if (!settings.enabled) {
      if (player?.running) { teardown(); SB.log('꺼짐 — 설정에서 껐다. 영상은 원래대로 재생된다'); }
      return;
    }
    if (!prev.enabled) { boot(); return; }        // 다시 켰다 — 붙일 때 설정을 건다
    if (player?.running) {
      const c = SB.applySettings(settings, { player, detector });
      if (c.length) SB.log(`설정 — ${c.join(' · ')}`);
    }
  }

  function teardown() {
    stopYamnet(); clip?.stop(); clip = null; visual?.stop(); detector?.stop(); player?.stop(); player = null;
  }

  function findVideo() {
    return document.querySelector('video.html5-main-video') ||
           document.querySelector('#movie_player video');
  }

  function attach(video) {
    if (player?.running) return;
    if (!video || video.readyState < 2) return;

    player = new SB.Player(video);
    try {
      player.start();
    } catch (e) {
      // start()는 캔버스를 붙이고 <video>를 숨긴 뒤 오디오를 잡는다. 중간에 터지면
      // 그 상태로 멈춰 화면이 까맣게 남는다. 원복해서 최소한 영상은 보이게 한다.
      console.error('[scareblock] 시작 실패 — 원복한다', e);
      try { player.stop(); } catch { /* 부분 초기화 상태일 수 있다 */ }
      player = null;
      window.scareblock = { error: String(e), stats: () => ({ 시작실패: String(e) }) };
      return;
    }
    // 기본은 규칙 탐지기. M1을 다시 잴 때처럼 10초 주기가 필요하면 config에서 'fake'로 바꾼다.
    detector = SB.config.DETECTOR === 'fake'
      ? new SB.FakeDetector(video, player)
      : new SB.RuleDetector(video, player);
    try {
      detector.start();
    } catch (e) {
      // 탐지기가 못 붙어도 지연 재생은 살린다 — 블러만 안 걸린다
      console.error('[scareblock] 탐지기 시작 실패 — 지연 재생만 한다', e);
    }
    // 시각 특징은 아직 트리거를 내지 않는다 — 기록만 한다. 못 붙어도 나머지는 그대로 돈다
    try {
      visual = new SB.VisualSampler(video);
      visual.start();
    } catch (e) {
      console.error('[scareblock] 시각 특징 시작 실패 — 오디오 탐지만 한다', e);
      visual = null;
    }

    window.scareblock = {
      stats: () => player.report(),
      measure: (on = true) => player.measure(on),
      renderTiming: () => player.renderTiming(),
      detections: () => detector.n,
      visual: () => visual?.report() ?? { 안내: '시각 특징이 꺼져 있다' },
      // 별도 문서 추론 상태 — 왕복 시간 · 실패 수 · 탭 구간
      // 말소리 기각 — 받음 · 통과 · 기각 · 판정 없이 통과한 수와 최근 판정
      speech: () => detector.gate?.report() ?? { 안내: '말소리 기각이 꺼져 있다(config.SPEECH_GATE)' },
      infer: () => ({ 클라이언트: infer?.report() ?? '꺼짐(config.YAMNET · CLIP)', 탭: tap?.report() ?? '안 붙음' }),
      // CLIP — 프레임 수 · 건너뜀 · 왕복 시간 · 최근 점수
      clip: () => clip?.report() ?? { 안내: 'CLIP이 안 붙었다 — 콘솔의 「CLIP 꺼짐」 줄을 본다' },
      // 확인용: 미디어 시각 t(기본 지금보다 1.5초 전)부터 패치 하나의 상위 5개 클래스와 Speech
      yamnet: (t) => yamnetTop(t ?? video.currentTime - 1.5),
      // 데모용 동작점 전환. 규칙 탐지기일 때만 있다 — E1 보고값은 20 그대로다
      setSurge: (n) => detector.setSurge?.(n),
      stop: () => teardown(),
      settings: () => ({ ...settings }),
      blurNow: (dur = 1.5) => player.addTriggers([{
        time: video.currentTime, category: 'manual',
        confidence: 1, duration: dur, source: 'manual',
      }]),
      player: () => player,
    };
    const applied = SB.applySettings(settings, { player, detector });
    if (applied.length) SB.log(`설정 — ${applied.join(' · ')}`);
    if (infer && SB.config.YAMNET) attachYamnet(video, player);
    if (infer && SB.config.CLIP) attachClip(video, player);
    SB.log('준비됨 — scareblock.stats() / scareblock.stop() / scareblock.blurNow()');
  }

  let names = null;
  async function yamnetTop(t) {
    if (!SB.yamnet) return { 안내: 'YAMNet이 안 붙었다 — scareblock.infer() 로 이유를 본다' };
    const s = await SB.yamnet.scoresAt(t);
    if (!s) return { 안내: `${t.toFixed(2)}s 패치를 못 얻었다 (지나갔거나 시크) — scareblock.infer()` };
    if (!names) names = (await infer.status())?.names || [];
    const top = [...s].map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, 5)
      .map(([i, v]) => `${names[i] ?? i} ${v.toFixed(3)}`);
    return { 시각: +t.toFixed(2), Speech: +s[0].toFixed(3), 상위5: top };
  }

  async function boot() {
    await settingsReady;
    if (!settings.enabled) return;
    // 콘텐츠 스크립트는 매치되는 URL로 **문서가 로드될 때만** 주입된다. /watch로만
    // 매치하면 홈·검색에서 영상을 클릭하는 SPA 경로에는 주입 자체가 없어 확장이
    // 없는 것과 같다. 그래서 youtube.com 전체로 매치하고 여기서 걸러낸다.
    if (location.pathname !== '/watch') return;
    const v = findVideo();
    if (!v) return;
    if (v.readyState >= 2) attach(v);
    else v.addEventListener('loadeddata', () => attach(v), { once: true });
  }

  // 유튜브 SPA 네비게이션
  document.addEventListener('yt-navigate-finish', () => {
    if (player?.running) teardown();
    setTimeout(boot, 800);
  });

  boot();
})();
