/**
 * 중계 — ISOLATED world 콘텐츠 스크립트. 페이지의 `infer-client.js`와 offscreen 문서 사이를 잇는다.
 *
 * 페이지 쪽(MAIN)은 `chrome.runtime`이 없고, offscreen 쪽은 페이지 window가 없다. 이 스크립트만 둘 다 본다.
 * 오디오는 window 메시지로는 Float32Array 그대로 건너오지만, chrome.runtime 메시지는 JSON이라 base64로 바꾼다.
 */
(() => {
  const TAG = 'scareblock-infer';
  const OPS = new Set(['status', 'yamnet', 'clipStatus', 'clip']);
  let ready = null;    // offscreen을 띄우는 약속 — 실패하면 다음 요청에서 다시 시도한다

  function ensureOffscreen() {
    if (!ready) {
      ready = chrome.runtime.sendMessage({ to: 'sb-sw', op: 'ensure-offscreen' }).then((r) => {
        if (!r?.ok) throw new Error(r?.error || 'offscreen을 띄우지 못했다');
      });
      ready.catch(() => { ready = null; });
    }
    return ready;
  }

  function reply(id, body) {
    window.postMessage({ tag: TAG, dir: 'res', id, ...body }, location.origin);
  }

  window.addEventListener('message', async (e) => {
    if (e.source !== window) return;
    const m = e.data;
    if (!m || m.tag !== TAG || m.dir !== 'req' || !Number.isInteger(m.id) || !OPS.has(m.op)) return;
    try {
      const msg = { to: 'sb-offscreen', op: m.op };
      if (m.op === 'yamnet') {
        if (!(m.pcm instanceof Float32Array)) throw new Error('pcm이 Float32Array가 아니다');
        msg.pcm = SB.pcm.encode(m.pcm);
      }
      if (m.op === 'clip') {
        if (!(m.rgb instanceof Uint8Array) || m.rgb.length !== 224 * 224 * 3) throw new Error('rgb가 224×224×3 Uint8Array가 아니다');
        msg.rgb = SB.pcm.encodeBytes(m.rgb);
      }
      await ensureOffscreen();
      const r = await chrome.runtime.sendMessage(msg);
      if (!r) throw new Error('offscreen 응답 없음');
      if (!r.ok) throw new Error(r.error);
      reply(m.id, { ok: true, result: r.result });
    } catch (err) {
      reply(m.id, { ok: false, error: String(err?.message || err) });
    }
  });
})();
