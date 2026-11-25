/**
 * 별도 문서 추론 경로 검증 — 브라우저 없이 가짜 window · chrome · ort로 잇는다.
 *   node extension/test/infer.test.js
 *
 * 페이지(infer-client) → 중계(bridge) → 서비스 워커(background) → offscreen 까지 메시지 모양이 맞는지,
 * 오디오 탭이 패치를 미디어 시각으로 기다려 주는지, 어디가 빠져도 null로 끝나 탐지를 막지 않는지 본다.
 * **실제 Chrome의 offscreen · wasm 로드는 여기서 보지 못한다** — README의 확인 절차로 본다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const v8 = require('v8');
const vm = require('vm');

const src = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');

// ── 가짜 window: 같은 window를 페이지와 중계가 같이 본다(실제로도 메시지는 같은 window로 오간다)
global.window = global;
global.location = { origin: 'https://www.youtube.com' };
const winListeners = [];
global.addEventListener = (t, fn) => { if (t === 'message') winListeners.push(fn); };
global.removeEventListener = (t, fn) => {
  const i = winListeners.indexOf(fn);
  if (i >= 0) winListeners.splice(i, 1);
};
global.postMessage = (data, origin) => {
  assert.strictEqual(origin, location.origin);
  const d = v8.deserialize(v8.serialize(data));   // 구조화 복제 — 형식 배열이 그대로 건넌다
  setTimeout(() => winListeners.slice().forEach((fn) => fn({ source: global, data: d })), 0);
};

// ── 가짜 chrome: 메시지는 모든 확장 문맥(서비스 워커 · offscreen)의 onMessage로 간다
const rtListeners = [];
const created = [];
global.chrome = {
  runtime: {
    getURL: (p) => `chrome-extension://sb/${p}`,
    onMessage: { addListener: (fn) => rtListeners.push(fn) },
    onInstalled: { addListener: () => {} },   // 첫 설치 안내 탭(설정 화면) — 여기서는 보지 않는다
    sendMessage: (msg) => new Promise((resolve, reject) => {
      const m = JSON.parse(JSON.stringify(msg));        // 실제처럼 JSON만 건넌다
      let async = false;
      for (const fn of rtListeners) if (fn(m, {}, resolve)) async = true;
      if (!async) reject(new Error('Could not establish connection. Receiving end does not exist.'));
    }),
    getContexts: async () => created.map((u) => ({ documentUrl: u })),
  },
  offscreen: {
    createDocument: async ({ url, reasons }) => {
      assert.deepStrictEqual(reasons, ['WORKERS']);
      created.push(url);
      eval(src('src/offscreen.js'));                    // 문서가 뜨면 리스너가 걸린다
    },
  },
};

// ── 가짜 ort: 점수 0번(Speech) = 입력 절댓값 평균, 1번 = 입력 길이. 넘어온 파형을 확인할 수 있게
let ortRuns = 0;
global.ort = {
  env: { wasm: {} },
  Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } },
  InferenceSession: {
    create: async (url) => {
      if (url.endsWith('models/clip_visual.onnx')) {
        // 가짜 CLIP: 임베딩 = [입력 첫 값, 0, 0, 1, 0…] — 정규화된 픽셀이 건너왔는지 본다
        return {
          inputNames: ['pixels'],
          outputNames: ['image_emb'],
          run: async (feeds) => {
            ortRuns++;
            assert.deepStrictEqual(feeds.pixels.dims, [1, 3, 224, 224]);
            const e = new Float32Array(512);
            e[0] = feeds.pixels.data[0];
            e[3] = 1;
            return { image_emb: { dims: [1, 512], data: e } };
          },
        };
      }
      assert.ok(url.endsWith('models/yamnet.onnx'), url);
      return {
        inputNames: ['waveform'],
        outputNames: ['scores', 'embeddings'],
        run: async (feeds, outs) => {
          ortRuns++;
          assert.deepStrictEqual(outs, ['scores']);
          const x = feeds.waveform.data;
          const data = new Float32Array(2 * 521);         // 패치 2개라고 해도 첫 줄만 써야 한다
          let s = 0;
          for (const v of x) s += Math.abs(v);
          data[0] = s / x.length;
          data[1] = x.length;
          data[521] = 99;
          return { scores: { dims: [2, 521], data } };
        },
      };
    },
  },
};
// 가짜 heads: 문장 임베딩이 축 벡터라 cos가 임베딩 성분 그대로다 (blood=0번 · ordinary=3번)
const unit = (i) => Array.from({ length: 512 }, (_, j) => (j === i ? 1 : 0));
const HEADS = { order: ['blood', 'syringe', 'spider', 'ordinary'], logit_scale: 100,
  text_emb: [unit(0), unit(1), unit(2), unit(3)], probe: { cat: 'blood', coef: unit(0), intercept: 0 } };
let headsMissing = false;
global.fetch = async (url) => (url.endsWith('clip_heads.json')
  ? { ok: !headsMissing, json: async () => HEADS }
  : { text: async () => 'index,mid,display_name\n0,/m/09x0r,Speech\n1,/m/x,"Child speech, kid speaking"\n' });

// offscreen.js가 이 전역에서 돈다 — clip-prep.js(SB.clipNormalize · clipHeads)가 있어야 한다 (offscreen.html과 같다)
for (const f of ['src/pcm.js', 'src/clip-prep.js', 'src/infer-client.js', 'src/audio-tap.js', 'background.js']) eval(src(f));

// 중계는 **따로 떨어진 전역**에서 돌린다 — 실제 ISOLATED world처럼 페이지 쪽 SB가 보이지 않는다.
// 같은 전역에서 돌리면 중계가 페이지의 SB에 기대도 테스트가 통과한다(11.23 실기에서 그렇게 놓쳤다).
const bridgeCtx = vm.createContext({
  document: { getElementById: () => null, createElement: () => ({ style: {}, setAttribute() {} }), body: { appendChild() {} } },   // CLIP 자르기 프레임을 넣는 자리 — 여기서는 보지 않는다
  window: null, location, chrome, btoa, console, setTimeout,
  Float32Array, Uint8Array, Number, Set, String, Error, Promise,   // 메시지로 건너온 배열과 같은 realm
  addEventListener: (...a) => global.addEventListener(...a),
  postMessage: (...a) => global.postMessage(...a),
});
const bridgeGlobal = vm.runInContext('globalThis', bridgeCtx);   // 컨텍스트 안에서 보이는 전역(프록시)
bridgeCtx.window = bridgeGlobal;
// 가짜 window 메시지의 source는 페이지 전역이다 — 중계의 `e.source !== window` 검사가 통과하도록 같은 것으로 본다
const bridgeSeen = (fn) => (e) => fn(e.source === global ? { ...e, source: bridgeGlobal } : e);
const keepAdd = global.addEventListener;
bridgeCtx.addEventListener = (t, fn) => keepAdd(t, bridgeSeen(fn));
vm.runInContext(src('src/bridge.js'), bridgeCtx);
SB.log = () => {};

let pass = 0, total = 0;
const tests = [];
const ok = (name, fn) => tests.push([name, fn]);

ok('status — 서비스 워커가 offscreen을 띄우고 모델을 올린다', async () => {
  const c = new SB.InferClient();
  const st = await c.status();
  assert.ok(st?.ready, JSON.stringify(st) + ' ' + c.report().마지막오류);
  assert.strictEqual(created.length, 1);
  assert.strictEqual(st.names[1], 'Child speech, kid speaking');
  assert.strictEqual(ort.env.wasm.numThreads, 1);
  c.dispose();
});

ok('yamnet — 파형이 손실 없이 건너가고 첫 패치 점수만 돌아온다', async () => {
  const c = new SB.InferClient();
  const pcm = Float32Array.from({ length: SB.YAMNET_WIN_N }, (_, i) => (i % 2 ? 0.25 : -0.25));
  const s = await c.yamnet(pcm);
  assert.ok(s instanceof Float32Array && s.length === 521);
  assert.ok(Math.abs(s[0] - 0.25) < 1e-7, `s[0]=${s[0]}`);
  assert.strictEqual(s[1], SB.YAMNET_WIN_N);
  assert.strictEqual(created.length, 1, 'offscreen은 한 번만 띄운다');
  assert.strictEqual(c.report().성공, 1);
  c.dispose();
});

ok('yamnet — 길이가 패치 하나가 아니면 보내기 전에 던진다', async () => {
  const c = new SB.InferClient();
  await assert.rejects(() => c.yamnet(new Float32Array(16000)), /15600/);
  c.dispose();
});

ok('offscreen이 실패를 답하면 null — 탐지 쪽은 「검증 못 함」으로 처리한다', async () => {
  const c = new SB.InferClient();
  const keep = ort.Tensor;
  ort.Tensor = class { constructor() { throw new Error('wasm 터짐'); } };
  const s = await c.yamnet(new Float32Array(SB.YAMNET_WIN_N));
  ort.Tensor = keep;
  assert.strictEqual(s, null);
  assert.match(c.report().마지막오류, /wasm 터짐/);
  c.dispose();
});

ok('clip — 자른 RGB가 바이트 그대로 건너가 정규화되고, 점수가 돌아온다', async () => {
  const c = new SB.InferClient();
  const st = await c.clipStatus();
  assert.ok(st?.ready && st.probe === 'blood', JSON.stringify(st));
  const rgb = new Uint8Array(224 * 224 * 3);
  rgb[0] = 255;                                       // 첫 픽셀 R
  const r = await c.clip(rgb);
  assert.ok(r, c.report().마지막오류);
  const x0 = (1 - SB.CLIP_MEAN[0]) / SB.CLIP_STD[0];   // offscreen이 정규화한 첫 값
  const n = Math.hypot(x0, 1);
  const zs = 1 / (1 + Math.exp(-100 * (x0 / n - 1 / n)));
  assert.ok(Math.abs(r.zs.blood - zs) < 1e-6, `zs.blood=${r.zs.blood} 기대 ${zs}`);
  assert.ok(Math.abs(r.zs.spider - 1 / (1 + Math.exp(100 / n))) < 1e-6);
  assert.ok(Math.abs(r.probe.blood - 1 / (1 + Math.exp(-x0 / n))) < 1e-6);
  await assert.rejects(() => c.clip(new Uint8Array(10)), /224/);
  c.dispose();
});

ok('clip — 중계는 모양이 틀린 rgb를 버리고 null로 끝난다', async () => {
  const c = new SB.InferClient();
  const r = await c.call('clip', { rgb: new Uint8Array(5) }, 500);
  assert.strictEqual(r, null);
  assert.match(c.report().마지막오류, /224×224×3/);
  c.dispose();
});

ok('중계가 없으면 시간 초과로 null', async () => {
  const quiet = {
    location, addEventListener() {}, removeEventListener() {}, postMessage() {},
  };
  const c = new SB.InferClient(quiet, 30);
  assert.strictEqual(await c.call('status', {}, 30), null);
  assert.strictEqual(await c.yamnet(new Float32Array(SB.YAMNET_WIN_N)), null);
  assert.strictEqual(c.report().시간초과, 2);
  c.dispose();
});

ok('중계는 페이지 쪽 SB 없이 돈다', () => {
  assert.strictEqual(vm.runInContext('typeof SB', bridgeCtx), 'undefined');
});

ok('중계는 모르는 op · 다른 출처 메시지를 무시한다', async () => {
  const before = ortRuns;
  let replies = 0;
  const spy = (e) => { if (e.data?.dir === 'res') replies++; };
  winListeners.push(spy);
  postMessage({ tag: SB.INFER_TAG, dir: 'req', id: 1, op: 'eval', code: 'x' }, location.origin);
  winListeners.forEach((fn) => fn({ source: {}, data: { tag: SB.INFER_TAG, dir: 'req', id: 2, op: 'status' } }));
  await new Promise((r) => setTimeout(r, 20));
  winListeners.splice(winListeners.indexOf(spy), 1);
  assert.strictEqual(replies, 0);
  assert.strictEqual(ortRuns, before);
});

// ── 오디오 탭: 가짜 오디오 그래프 · <video>
function fakeAv(sr = 48000) {
  const vl = {};
  const video = {
    currentTime: 50, playbackRate: 1, paused: false, seeking: false,
    addEventListener: (t, fn) => { (vl[t] ||= []).push(fn); },
    removeEventListener: (t, fn) => { vl[t] = (vl[t] || []).filter((f) => f !== fn); },
    fire: (t) => (vl[t] || []).forEach((fn) => fn()),
  };
  let node = null;
  const ac = {
    sampleRate: sr, destination: {},
    createScriptProcessor: () => (node = { connect() {}, disconnect() {} }),
  };
  const player = { audio: { ac, src: { connect() {}, disconnect() {} } } };
  const feed = (sec, val = 0.1) => {
    const n = 2048;
    const buf = new Float32Array(n).fill(val);
    for (let k = 0; k < Math.round((sec * sr) / n); k++) {
      video.currentTime += n / sr;
      node.onaudioprocess({ inputBuffer: { duration: n / sr, numberOfChannels: 1, getChannelData: () => buf } });
    }
  };
  return { video, player, feed };
}

ok('탭 — 아직 안 들어온 패치는 기다렸다가 받는다', async () => {
  const { video, player, feed } = fakeAv();
  const tap = new SB.AudioTap(video, player);
  tap.start();
  feed(0.5);
  const p = tap.patchAt(50.2);
  assert.strictEqual(tap.waiters.length, 1);
  feed(1.5);
  const pcm = await p;
  assert.ok(pcm && pcm.length === SB.YAMNET_WIN_N);
  assert.ok(Math.abs(pcm[8000] - 0.1) < 1e-4, `${pcm[8000]}`);
  assert.strictEqual(tap.waiters.length, 0);
  assert.ok(tap.report().콜백평균_ms !== null);
  tap.stop();
});

ok('탭 — 기다리는 중에 시크하면 null', async () => {
  const { video, player, feed } = fakeAv();
  const tap = new SB.AudioTap(video, player);
  tap.start();
  feed(0.5);
  const p = tap.patchAt(50.3);
  video.fire('seeked');
  assert.strictEqual(await p, null);
  tap.stop();
});

ok('탭 — 일시정지 뒤 재생은 리셋하지 않고, 버퍼링을 거친 재생만 리셋한다', async () => {
  const { video, player, feed } = fakeAv();
  const tap = new SB.AudioTap(video, player);
  tap.start();
  feed(0.5);
  const g = tap.core.gen;
  video.fire('playing');
  assert.strictEqual(tap.core.gen, g);
  video.fire('waiting');
  video.fire('playing');
  assert.notStrictEqual(tap.core.gen, g);
  tap.stop();
});

ok('탭 — 시간 안에 안 들어오면 null', async () => {
  const { video, player, feed } = fakeAv();
  const tap = new SB.AudioTap(video, player);
  tap.start();
  feed(0.2);
  assert.strictEqual(await tap.patchAt(60, 20), null);
  tap.stop();
});

ok('SB.Yamnet — 미디어 시각 → 패치 → offscreen 점수까지 한 번에', async () => {
  const { video, player, feed } = fakeAv(44100);
  const tap = new SB.AudioTap(video, player);
  tap.start();
  const c = new SB.InferClient();
  const y = new SB.Yamnet(tap, c);
  feed(2.5, -0.3);
  const s = await y.scoresAt(50.5);
  assert.ok(s, c.report().마지막오류);
  assert.ok(Math.abs(s[0] - 0.3) < 1e-3, `Speech 자리=${s[0]}`);
  tap.stop();
  c.dispose();
});

(async () => {
  console.log('별도 문서 추론');
  for (const [name, fn] of tests) {
    total++;
    try { await fn(); console.log(`  ✓ ${name}`); pass++; }
    catch (e) { console.error(`  ✗ ${name}\n    ${e.stack}`); process.exitCode = 1; }
  }
  console.log(`${pass}/${total} 통과`);
})();
