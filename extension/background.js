/**
 * 서비스 워커 — offscreen 문서를 띄우는 일만 한다. 추론 메시지는 거치지 않는다
 * (콘텐츠 스크립트의 chrome.runtime.sendMessage는 offscreen 문서에도 바로 닿는다).
 */
const OFFSCREEN = 'offscreen.html';
let creating = null;

async function ensureOffscreen() {
  const url = chrome.runtime.getURL(OFFSCREEN);
  const have = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'], documentUrls: [url],
  });
  if (have.length) return;
  if (!creating) {
    creating = chrome.offscreen.createDocument({
      url: OFFSCREEN,
      reasons: ['WORKERS'],
      justification: 'YAMNet(ONNX · wasm) 추론 — 유튜브 페이지 CSP 안에서는 wasm을 올릴 수 없다',
    }).finally(() => { creating = null; });
  }
  await creating;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.to !== 'sb-sw') return false;
  if (msg.op === 'ensure-offscreen') {
    ensureOffscreen().then(
      () => sendResponse({ ok: true }),
      (e) => sendResponse({ ok: false, error: String(e?.message || e) }),
    );
    return true;   // 비동기 응답
  }
  return false;
});

// 첫 설치 때 안내 탭을 연다 — 안정적인 볼륨 끄기 · 3초 지연 · 설정 위치를 알려 준다(settings.html ?welcome)
chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === 'install') chrome.tabs.create({ url: chrome.runtime.getURL('settings.html?welcome') });
});
