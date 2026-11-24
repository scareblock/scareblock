/**
 * CLIP 자르기 프레임 — 유튜브 페이지 안에 숨겨 둔 확장 문서(`clip-frame.html`).
 *
 * **왜 프레임인가 (11.24 실기).** 1080p 프레임을 원본 크기로 읽어 PIL과 같은 방식으로 자르면 메인 스레드에서
 * 약 66 ms가 든다 — CLIP을 켜면 1초마다 렌더가 끊겨 M1이 깨졌다(127초에 127번). 유튜브 CSP는 `worker-src`로
 * blob · data Worker를 모두 막는다. 출처가 다른 프레임은 Chrome이 다른 프로세스에서 돌리므로, 여기서 자르면
 * 유튜브 메인 스레드가 멈추지 않는다(페이지 쪽에는 createImageBitmap 0.3 ms만 남는다. 자르는 동안 페이지의
 * 32 ms 넘는 rAF 간격 수는 자르지 않을 때와 같았다 — 5/300 vs 4/228).
 *
 * 계산은 `clip-prep.js` 그대로(SB.rgbaToRgb · SB.clipCrop)라 페이지에서 자르던 것과 **바이트가 같다** — θ를 다시 볼 일이 없다.
 * 자른 결과는 페이지를 거치지 않고 여기서 바로 offscreen에 보내고, 점수만 페이지로 돌려준다.
 *
 *   페이지 ─ImageBitmap(옮김)─▶ 이 프레임: 그리기 · 읽기 · 자르기 ─chrome.runtime─▶ offscreen: CLIP ─▶ 점수 ─▶ 페이지
 */
(() => {
  const PARENT = 'https://www.youtube.com';
  const TAG = 'sb-clipframe';
  let canvas = null, ctx = null;
  let ready = null;   // offscreen 띄우기 — 실패하면 다음 요청에서 다시

  function ensureOffscreen() {
    if (!ready) {
      ready = chrome.runtime.sendMessage({ to: 'sb-sw', op: 'ensure-offscreen' }).then((r) => {
        if (!r?.ok) throw new Error(r?.error || 'offscreen을 띄우지 못했다');
      });
      ready.catch(() => { ready = null; });
    }
    return ready;
  }

  /** ImageBitmap → 자른 RGB [224·224·3]. 끝나면 비트맵을 닫는다 */
  function crop(bitmap) {
    const w = bitmap.width, h = bitmap.height;
    if (!canvas || canvas.width !== w || canvas.height !== h) {
      canvas = new OffscreenCanvas(w, h);
      ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
    }
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    return SB.clipCrop(SB.rgbaToRgb(ctx.getImageData(0, 0, w, h).data), w, h);
  }

  window.addEventListener('message', async (e) => {
    if (e.origin !== PARENT || e.source !== parent || e.data?.tag !== TAG) return;
    const { op, id } = e.data;
    const reply = (body, transfer) => parent.postMessage({ tag: TAG, op, id, ...body }, PARENT, transfer || []);
    if (op === 'ping') { reply({ ok: true }); return; }
    if (op !== 'clip' && op !== 'crop') return;
    const bitmap = e.data.bitmap;
    if (!(bitmap instanceof ImageBitmap)) { reply({ ok: false, error: 'ImageBitmap이 아니다' }); return; }
    try {
      const t0 = performance.now();
      const rgb = crop(bitmap);
      const cropMs = performance.now() - t0;
      if (op === 'crop') { reply({ ok: true, cropMs, crop: rgb }, [rgb.buffer]); return; }   // 확인용
      await ensureOffscreen();
      const r = await chrome.runtime.sendMessage({ to: 'sb-offscreen', op: 'clip', rgb: SB.pcm.encodeBytes(rgb) });
      if (!r) throw new Error('offscreen 응답 없음');
      if (!r.ok) throw new Error(r.error);
      reply({ ok: true, cropMs, result: r.result });
    } catch (err) {
      try { bitmap.close(); } catch { /* 이미 닫힘 */ }
      reply({ ok: false, error: String(err?.message || err) });
    }
  });
})();
