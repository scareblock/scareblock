/**
 * 중계 — ISOLATED world 콘텐츠 스크립트. 페이지의 `infer-client.js`와 offscreen 문서 사이를 잇는다.
 *
 * 페이지 쪽(MAIN)은 `chrome.runtime`이 없고, offscreen 쪽은 페이지 window가 없다. 이 스크립트만 둘 다 본다.
 * 오디오는 window 메시지로는 Float32Array 그대로 건너오지만, chrome.runtime 메시지는 JSON이라 base64로 바꾼다.
 *
 * **다른 파일에 기대지 않는다.** 처음에는 `pcm.js`를 이 항목에도 넣어 `SB.pcm.encode`를 썼는데, 같은 파일이
 * MAIN 항목에도 있어 이 world에는 주입되지 않은 것으로 보인다 — 추론 요청 417건이 전부 `SB is not defined`로 실패했다
 * (11.23 실기). 그래서 인코딩을 여기 따로 둔다. 바이트 배치는 `SB.pcm.encode`와 같고, offscreen이 `SB.pcm.decode`로 푼다.
 * CLIP 프레임(224×224×3 RGB 바이트)도 같은 인코더로 보낸다 — offscreen이 `SB.pcm.decodeBytes`로 푼다(#54 리뷰 🔴1).
 */
(() => {
  const TAG = 'scareblock-infer';

  /** Uint8Array → base64 — `pcm.js` SB.pcm.encodeBytes와 같은 출력 */
  function encodeBytes(u8) {
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    }
    return btoa(s);
  }

  /** Float32Array → base64 (리틀 엔디언 바이트 그대로) — `pcm.js` SB.pcm.encode와 같은 출력 */
  const encodePcm = (f32) => encodeBytes(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength));
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
        msg.pcm = encodePcm(m.pcm);
      }
      if (m.op === 'clip') {
        if (!(m.rgb instanceof Uint8Array) || m.rgb.length !== 224 * 224 * 3) throw new Error('rgb가 224×224×3 Uint8Array가 아니다');
        msg.rgb = encodeBytes(m.rgb);
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

  // ── 설정 ── chrome.storage.local의 settings를 페이지에 넘긴다(settings.js). 페이지는 처음에 한 번 달라고 하고,
  // 바뀌면 여기서 다시 보낸다. 값 검사는 페이지(SB.normalizeSettings)가 한다 — 여기는 그대로 옮기기만 한다
  const STAG = 'scareblock-settings';
  async function pushSettings() {
    let settings = null;
    try { ({ settings } = await chrome.storage.local.get('settings')); } catch { /* 저장소를 못 읽으면 페이지는 기본값 */ }
    window.postMessage({ tag: STAG, dir: 'push', settings: settings ?? null }, location.origin);
  }
  window.addEventListener('message', (e) => {
    if (e.source === window && e.data?.tag === STAG && e.data.dir === 'get') pushSettings();
  });
  chrome.storage?.onChanged?.addListener((changes, area) => {
    if (area === 'local' && changes.settings) pushSettings();
  });
  pushSettings();
})();
