/**
 * CLIP 자르기를 확장 프레임으로 옮긴 길 검증 — 페이지 쪽(ClipFrameClient · ClipSampler)과 프레임 쪽(clip-frame.js).
 *   node extension/test/clip-frame.test.js
 *
 * 실제 다른 프로세스 · ImageBitmap은 여기서 못 만든다. 메시지 모양, 옮기기(transfer), 시간 초과 · 실패가 null로 끝나는지,
 * 프레임이 페이지와 **같은 계산**(clip-prep.js)으로 자르는지를 본다. 렌더 끊김은 실기로 본다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

const src = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
global.window = global;
global.location = { origin: 'https://www.youtube.com' };
for (const f of ['src/config.js', 'src/samples.js', 'src/pcm.js', 'src/clip-prep.js', 'src/clip-sampler.js']) eval(src(f));
SB.log = () => {};

let pass = 0, total = 0;
const tests = [];
const ok = (name, fn) => tests.push([name, fn]);
const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));

/** 가짜 ImageBitmap — 프레임 쪽 OffscreenCanvas가 읽을 RGBA를 들고 있다 */
class FakeBitmap {
  constructor(w, h, fill) { this.width = w; this.height = h; this.rgba = new Uint8ClampedArray(w * h * 4).fill(fill); this.closed = false; }
  close() { this.closed = true; }
}

/** 페이지 window ↔ 프레임 window를 잇는다. 프레임은 떨어진 vm에서 clip-frame.js를 돈다 */
function world({ offscreen } = {}) {
  const pageLs = [];
  const frameLs = [];
  const sent = [];     // 프레임이 chrome.runtime으로 보낸 것
  const frameWin = {};
  const pageWin = {
    location, addEventListener: (t, f) => { if (t === 'message') pageLs.push(f); },
    removeEventListener: (t, f) => { const i = pageLs.indexOf(f); if (i >= 0) pageLs.splice(i, 1); },
    postMessage: (d) => { if (d?.op === 'need') frameEl.attached = true; },   // 중계 역할: 「달라」면 붙인다
  };
  // 프레임 → 페이지
  const parentProxy = {
    postMessage: (d, origin, transfer) => {
      assert.strictEqual(origin, 'https://www.youtube.com');
      setTimeout(() => pageLs.forEach((f) => f({ source: frameWin, data: d })), 0);
    },
  };
  const ctx = vm.createContext({
    console, setTimeout, Promise, Math, Uint8Array, Uint8ClampedArray, Float32Array, Number, String, Error, btoa,
    performance, ImageBitmap: FakeBitmap,
    OffscreenCanvas: class {
      constructor(w, h) { this.width = w; this.height = h; }
      getContext() {
        let last = null;
        return { drawImage: (b) => { last = b.rgba; }, getImageData: (_x, _y, w, h) => ({ data: last, width: w, height: h }) };
      }
    },
    chrome: {
      runtime: {
        sendMessage: async (m) => {
          sent.push(m);
          if (m.to === 'sb-sw') return { ok: true };
          return offscreen ? offscreen(m) : { ok: true, result: { zs: { blood: 0.9, spider: 0.01, syringe: 0.01 }, probe: {}, ms: 200 } };
        },
      },
    },
    addEventListener: (t, f) => { if (t === 'message') frameLs.push(f); },
  });
  const g = vm.runInContext('globalThis', ctx);
  ctx.window = g;
  ctx.parent = parentProxy;
  for (const f of ['src/pcm.js', 'src/clip-prep.js', 'src/clip-frame.js']) vm.runInContext(src(f), ctx);
  // 페이지 → 프레임
  frameWin.postMessage = (d, origin, transfer) => {
    assert.ok(origin.startsWith('chrome-extension://'), origin);
    if (d.bitmap) assert.ok(transfer.includes(d.bitmap), 'ImageBitmap은 옮겨야 한다(복사 아님)');
    setTimeout(() => frameLs.forEach((f) => f({ origin: 'https://www.youtube.com', source: parentProxy, data: d })), 0);
  };
  const frameEl = { attached: false, src: 'chrome-extension://abc/clip-frame.html', contentWindow: frameWin };
  const doc = { getElementById: (id) => (id === 'scareblock-clip-frame' && frameEl.attached ? frameEl : null) };
  return { pageWin, doc, sent, ctx, frameEl, frameLs, parentProxy };
}

ok('프레임을 달라고 하고, 붙으면 ping으로 준비를 확인한다', async () => {
  const w = world();
  const c = new SB.ClipFrameClient(w.pageWin, w.doc);
  assert.strictEqual(await c.ready(2000), true);
  assert.strictEqual(w.frameEl.attached, true);
});

ok('clip — 프레임이 자르고 offscreen에 바로 보내고, 점수만 돌아온다', async () => {
  const w = world();
  const c = new SB.ClipFrameClient(w.pageWin, w.doc);
  await c.ready(2000);
  const bm = new FakeBitmap(320, 180, 7);
  const r = await c.clip(bm);
  assert.ok(r, c.lastError);
  assert.strictEqual(r.zs.blood, 0.9);
  assert.ok(typeof r.cropMs === 'number');
  assert.ok(bm.closed, '프레임이 다 쓴 비트맵을 닫는다');
  const msg = w.sent.find((m) => m.to === 'sb-offscreen');
  assert.strictEqual(msg.op, 'clip');
  // 프레임이 보낸 바이트 = 페이지에서 clip-prep으로 자른 바이트 (같은 계산)
  const sentBytes = SB.pcm.decodeBytes(msg.rgb);
  const rgba = new Uint8ClampedArray(320 * 180 * 4).fill(7);
  const mine = SB.clipCrop(SB.rgbaToRgb(rgba), 320, 180);
  assert.strictEqual(sentBytes.length, 224 * 224 * 3);
  assert.deepStrictEqual(Array.from(sentBytes), Array.from(mine));
});

ok('offscreen이 실패하면 null — 그 장은 가리지 않는다', async () => {
  const w = world({ offscreen: () => ({ ok: false, error: 'wasm 터짐' }) });
  const c = new SB.ClipFrameClient(w.pageWin, w.doc);
  await c.ready(2000);
  assert.strictEqual(await c.clip(new FakeBitmap(64, 64, 1)), null);
  assert.match(c.lastError, /wasm 터짐/);
});

ok('프레임이 안 붙으면 ready가 false, clip은 null', async () => {
  const w = world();
  w.pageWin.postMessage = () => {};          // 중계가 없다
  const c = new SB.ClipFrameClient(w.pageWin, w.doc);
  assert.strictEqual(await c.ready(300), false);
  assert.match(c.lastError, /안 붙었다/);
  const bm = new FakeBitmap(8, 8, 0);
  assert.strictEqual(await c.clip(bm), null);
  assert.ok(bm.closed);
});

ok('프레임은 유튜브가 아닌 출처 · 부모가 아닌 창의 메시지를 무시한다', async () => {
  const w = world();
  const ask = (origin, source) => w.frameLs.forEach((f) => f({ origin, source, data: { tag: 'sb-clipframe', op: 'ping', id: 1 } }));
  let replies = 0;
  w.parentProxy.postMessage = () => { replies++; };
  ask('https://evil.example', w.parentProxy);
  ask('https://www.youtube.com', {});
  await tick(10);
  assert.strictEqual(replies, 0);
  ask('https://www.youtube.com', w.parentProxy);
  await tick(10);
  assert.strictEqual(replies, 1);
});

ok('샘플러 — 페이지는 비트맵만 떠서 넘기고, 점수로 트리거를 낸다', async () => {
  const triggers = [];
  const video = { videoWidth: 1920, videoHeight: 1080, currentTime: 100, requestVideoFrameCallback() {} };
  const player = { addTriggers: (t) => triggers.push(...t) };
  let given = null;
  global.createImageBitmap = async (v) => { assert.strictEqual(v, video); return (given = new FakeBitmap(4, 4, 0)); };
  const client = { lastError: null, clip: async (b) => { assert.strictEqual(b, given); return { zs: { blood: 0.9, spider: 0, syringe: 0 }, probe: {}, cropMs: 61 }; } };
  const s = new SB.ClipSampler(video, player, client, { ...SB.config, CLIP_CATS: ['blood', 'spider', 'syringe'] });
  s.running = true;
  await s._sample(99);
  assert.strictEqual(s.busy, false);
  assert.strictEqual(triggers.length, 1);
  assert.strictEqual(triggers[0].category, 'blood');
  const r = s.report();
  assert.strictEqual(r.프레임, 1);
  assert.strictEqual(r.프레임_자르기_ms.n, 1);
});

ok('샘플러 — 응답이 없으면 가리지 않고 응답없음으로 센다', async () => {
  const triggers = [];
  const video = { videoWidth: 640, videoHeight: 360, currentTime: 10, requestVideoFrameCallback() {} };
  global.createImageBitmap = async () => new FakeBitmap(4, 4, 0);
  const s = new SB.ClipSampler(video, { addTriggers: (t) => triggers.push(...t) }, { lastError: '시간 초과', clip: async () => null });
  s.running = true;
  await s._sample(9);
  assert.strictEqual(triggers.length, 0);
  assert.strictEqual(s.report().응답없음, 1);
  assert.strictEqual(s.report().마지막오류, '시간 초과');
});

ok('샘플러 — 비트맵을 못 뜨면 실패로 세고, 5번 연속이면 끈다', async () => {
  const video = { videoWidth: 640, videoHeight: 360, currentTime: 10, requestVideoFrameCallback() {} };
  global.createImageBitmap = async () => { throw new Error('InvalidStateError'); };
  const s = new SB.ClipSampler(video, { addTriggers() {} }, { clip: async () => null });
  s.running = true;
  for (let i = 0; i < 5; i++) await s._sample(i).catch((e) => s._fail(e));
  assert.strictEqual(s.busy, false);
  assert.strictEqual(s.running, false);
});

ok('중계 — 「달라」고 할 때만 프레임을 넣는다', () => {
  const added = [];
  const ls = [];
  const ctx = vm.createContext({
    location, btoa, console, setTimeout, Promise, Uint8Array, Float32Array, Number, Set, String, Error,
    chrome: { runtime: { getURL: (p) => `chrome-extension://abc/${p}`, sendMessage: async () => ({ ok: true }) },
      storage: { local: { get: async () => ({}) } } },
    document: {
      getElementById: () => (added.length ? added[0] : null),
      createElement: () => ({ style: {}, setAttribute() {} }),
      body: { appendChild: (f) => added.push(f) },
    },
    addEventListener: (t, f) => { if (t === 'message') ls.push(f); },
    postMessage: () => {},
  });
  const g = vm.runInContext('globalThis', ctx);
  ctx.window = g;
  vm.runInContext(src('src/bridge.js'), ctx);
  assert.strictEqual(added.length, 0, 'CLIP을 안 켜면 프레임도 없다');
  ls.forEach((f) => f({ source: g, data: { tag: 'sb-clipframe', op: 'need' } }));
  ls.forEach((f) => f({ source: g, data: { tag: 'sb-clipframe', op: 'need' } }));
  assert.strictEqual(added.length, 1, '한 번만 넣는다');
  assert.strictEqual(added[0].src, 'chrome-extension://abc/clip-frame.html');
  assert.strictEqual(added[0].id, 'scareblock-clip-frame');
});

(async () => {
  console.log('CLIP 자르기 프레임');
  for (const [name, fn] of tests) {
    total++;
    try { await fn(); console.log(`  ✓ ${name}`); pass++; }
    catch (e) { console.error(`  ✗ ${name}\n    ${e.stack.split('\n').slice(0, 3).join('\n    ')}`); process.exitCode = 1; }
  }
  console.log(`${pass}/${total} 통과`);
})();
