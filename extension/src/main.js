/** 조립. 유튜브가 SPA라 영상이 바뀌어도 따라붙어야 한다. */
window.SB = window.SB || {};

(() => {
  let player = null;
  let detector = null;

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
    detector = new SB.FakeDetector(video, player);   // 02단계에서 교체된다
    detector.start();

    window.scareblock = {
      stats: () => player.report(),
      measure: (on = true) => player.measure(on),
      renderTiming: () => player.renderTiming(),
      stop: () => { detector.stop(); player.stop(); player = null; },
      blurNow: (dur = 1.5) => player.addTriggers([{
        time: video.currentTime, category: 'manual',
        confidence: 1, duration: dur, source: 'manual',
      }]),
      player: () => player,
    };
    SB.log('준비됨 — scareblock.stats() / scareblock.stop() / scareblock.blurNow()');
  }

  function boot() {
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
    if (player?.running) { detector?.stop(); player.stop(); player = null; }
    setTimeout(boot, 800);
  });

  boot();
})();
