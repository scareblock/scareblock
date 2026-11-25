/**
 * #64 리뷰 🟡1 확인용: 예전 방식(페이지 캔버스에 drawImage(video))과 새 방식(createImageBitmap을 확장 프레임 `crop` op로 보냄)으로
 * 같은 프레임을 떠서 224×224×3 결과가 바이트 단위로 같은지 본다.
 *
 * 가운데 단계 하나를 더 본다: 같은 ImageBitmap을 페이지 캔버스에서 자른 것.
 *   A 예전   video → 페이지 캔버스 → clipCrop
 *   B 중간   createImageBitmap(video) → 페이지 캔버스 → clipCrop      (뜨는 방식만 바뀜)
 *   C 새     createImageBitmap(video) → 확장 프레임 → clipCrop         (뜨는 방식 + 다른 프로세스)
 * A≠B이면 뜨는 방식(createImageBitmap) 탓이고, B≠C이면 프레임 쪽 OffscreenCanvas 탓이다.
 *
 * 쓰는 법: 유튜브 영상 페이지, CLIP 켜짐(콘솔에 「CLIP 준비」가 찍힌 뒤), DevTools 콘솔 컨텍스트는 top.
 * 이 파일 전체를 붙여 넣으면 지금 장면에서 한 번 돈다. 그 뒤 다른 장면에서 `await sbParity()`를 다시 부른다.
 * 비교하는 동안 영상을 잠깐 멈췄다가 원래 상태로 돌려놓는다.
 */
(() => {
  const TAG = 'sb-clipframe';
  let seq = 0;

  function cropOnPage(src, w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    // 예전 ClipSampler와 같은 옵션
    const ctx = c.getContext('2d', { willReadFrequently: true, alpha: false });
    ctx.drawImage(src, 0, 0, w, h);
    return SB.clipCrop(SB.rgbaToRgb(ctx.getImageData(0, 0, w, h).data), w, h);
  }

  function cropInFrame(frame, bitmap) {
    const id = `parity-${++seq}`;
    const origin = frame.src.match(/^[a-z][a-z0-9+.-]*:\/\/[^/]+/i)[0];
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { window.removeEventListener('message', on); reject(new Error('프레임 응답 없음 (3 s)')); }, 3000);
      function on(e) {
        if (e.source !== frame.contentWindow || e.data?.tag !== TAG || e.data.id !== id) return;
        window.removeEventListener('message', on);
        clearTimeout(timer);
        e.data.ok ? resolve(e.data.crop) : reject(new Error(e.data.error));
      }
      window.addEventListener('message', on);
      frame.contentWindow.postMessage({ tag: TAG, op: 'crop', id, bitmap }, origin, [bitmap]);
    });
  }

  function diff(x, y) {
    if (x.length !== y.length) return { 같음: false, 길이: [x.length, y.length] };
    let n = 0, max = 0, sum = 0;
    for (let i = 0; i < x.length; i++) {
      const d = Math.abs(x[i] - y[i]);
      if (d) { n++; sum += d; if (d > max) max = d; }
    }
    return { 같음: n === 0, 다른_바이트: n, 비율: +(n / x.length).toFixed(5), 최대차: max, 평균차: n ? +(sum / n).toFixed(2) : 0 };
  }

  window.sbParity = async () => {
    if (!window.SB?.clipCrop) throw new Error('SB가 없다 — 콘솔 컨텍스트가 top인지, 확장이 켜졌는지 확인');
    const video = document.querySelector('video.html5-main-video') || document.querySelector('#movie_player video');
    const frame = document.getElementById('scareblock-clip-frame');
    if (!video?.videoWidth) throw new Error('<video>가 없거나 아직 프레임이 없다');
    if (!frame?.contentWindow) throw new Error('확장 프레임이 없다 — CLIP이 켜졌는지(「CLIP 준비」) 확인');

    const wasPaused = video.paused;
    video.pause();
    await new Promise((r) => setTimeout(r, 400));   // 멈춘 프레임이 자리 잡을 때까지

    const w = video.videoWidth, h = video.videoHeight, t = video.currentTime;
    try {
      const A = cropOnPage(video, w, h);
      const bmpB = await createImageBitmap(video);
      const bmpC = await createImageBitmap(video);
      const B = cropOnPage(bmpB, w, h);
      bmpB.close();
      const C = await cropInFrame(frame, bmpC);

      const out = {
        시각: +t.toFixed(3), 크기: `${w}×${h}`,
        'A예전 vs C새': diff(A, C),
        'A예전 vs B중간 (뜨는 방식)': diff(A, B),
        'B중간 vs C새 (프레임 쪽)': diff(B, C),
      };
      console.log(out['A예전 vs C새'].같음 ? '✅ 예전 = 새 (바이트가 같다)' : '❌ 예전 ≠ 새', out);
      return out;
    } finally {
      if (!wasPaused) video.play().catch(() => {});
    }
  };

  return sbParity();
})();
