/**
 * offscreen 줄 나누기 · CLIP 전용 Worker 검증 (#54 리뷰 🟡2).
 *   node extension/test/offscreen.test.js
 *
 * 가짜 chrome · ort · Worker로 offscreen.js와 clip-worker.js를 돌린다. 실제 스레드 분리는 Chrome에서만 보인다 —
 * 여기서는 줄이 모델마다 따로인지, Worker로 가는 길과 물러나는 길이 맞는지, 계산 결과가 예전 길과 같은지를 본다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

const src = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');

let pass = 0, total = 0;
const tests = [];
const ok = (name, fn) => tests.push([name, fn]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const unit = (i) => Array.from({ length: 512 }, (_, j) => (j === i ? 1 : 0));
const HEADS = { order: ['blood', 'syringe', 'spider', 'ordinary'], logit_scale: 100,
  text_emb: [unit(0), unit(1), unit(2), unit(3)], probe: { cat: 'blood', coef: unit(0), intercept: 0 } };

/** 가짜 ort — CLIP 한 장에 clipMs가 걸린다. 임베딩 = [입력 첫 값, 0, 0, 1, …] */
function fakeOrt(clipMs, log) {
  return {
    env: { wasm: {} },
    Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } },
    InferenceSession: {
      create: async (url) => {
        if (String(url).endsWith('clip_visual.onnx')) {
          return {
            inputNames: ['pixels'], outputNames: ['emb'],
            run: async (feeds) => {
              log.push('clip-start'); await sleep(clipMs); log.push('clip-end');
              const e = new Float32Array(512); e[0] = feeds.pixels.data[0]; e[3] = 1;
              return { emb: { dims: [1, 512], data: e } };
            },
          };
        }
        return {
          inputNames: ['waveform'], outputNames: ['scores'],
          run: async () => { log.push('yamnet'); return { scores: { dims: [1, 521], data: new Float32Array(521) } }; },
        };
      },
    },
  };
}

/**
 * offscreen 문서 하나를 만든다. worker: 'ok' | 'fail' | 'none'(Worker 없음)
 * 'ok'면 가짜 Worker가 clip-worker.js를 떨어진 vm에서 돌린다(자기 ort를 가진다 — 다른 스레드인 셈)
 */
function offscreenDoc({ worker = 'none', clipMs = 0, headsOk = true } = {}) {
  const log = [];
  const listeners = [];
  const transfers = [];
  const fetchImpl = async (url) => (String(url).endsWith('clip_heads.json')
    ? { ok: headsOk, json: async () => HEADS }
    : { ok: true, text: async () => 'index,mid,display_name\n0,/m/09x0r,Speech\n' });

  class FakeWorker {
    constructor(url) {
      if (worker === 'fail') throw new Error('SecurityError');
      assert.ok(url.endsWith('src/clip-worker.js'), url);
      this.onmessage = null;
      this.onerror = null;
      const self = {};
      const ctx = vm.createContext({
        console, setTimeout, Promise, Math, Float32Array, Uint8Array, Number, String, Error, performance, atob, btoa,
        fetch: fetchImpl, ort: fakeOrt(clipMs, log),
        postMessage: (m) => setTimeout(() => this.onmessage?.({ data: m }), 0),
        importScripts: (...fs2) => {
          assert.deepStrictEqual(fs2, ['../vendor/ort.wasm.min.js', 'pcm.js', 'clip-prep.js']);
          for (const f of ['src/pcm.js', 'src/clip-prep.js']) vm.runInContext(src(f), ctx);
        },
      });
      const g = vm.runInContext('globalThis', ctx);
      ctx.self = g;
      vm.runInContext(src('src/clip-worker.js'), ctx);
      this._g = g;
      Object.assign(self, g);
    }
    postMessage(m, transfer = []) {
      transfers.push(...transfer);
      const copy = m.rgb ? { ...m, rgb: new Uint8Array(m.rgb) } : m;   // 다른 스레드로 건너간다
      setTimeout(() => this._g.onmessage({ data: copy }), 0);
    }
    terminate() {}
  }

  const ctx = vm.createContext({
    console: { warn() {}, log() {} }, setTimeout, Promise, Math, Float32Array, Uint8Array, Number, String, Error,
    performance, atob, btoa, fetch: fetchImpl, ort: fakeOrt(clipMs, log),
    chrome: {
      runtime: {
        getURL: (p) => `chrome-extension://sb/${p}`,
        onMessage: { addListener: (f) => listeners.push(f) },
      },
    },
    ...(worker === 'none' ? {} : { Worker: FakeWorker }),
  });
  const g = vm.runInContext('globalThis', ctx);
  ctx.window = g;
  for (const f of ['src/pcm.js', 'src/clip-prep.js', 'src/offscreen.js']) vm.runInContext(src(f), ctx);
  const SB = vm.runInContext('SB', ctx);

  const send = (msg) => new Promise((resolve) => {
    let handled = false;
    for (const f of listeners) if (f(msg, {}, resolve)) handled = true;
    if (!handled) resolve(undefined);
  });
  const rgb = () => { const u = new Uint8Array(224 * 224 * 3); u[0] = 255; return SB.pcm.encodeBytes(u); };
  const pcm = () => SB.pcm.encode(new Float32Array(SB.YAMNET_WIN_N));
  return { send, log, transfers, rgb, pcm, SB };
}

for (const mode of ['none', 'ok']) {
  const name = mode === 'ok' ? 'Worker' : 'Worker 없음(이 문서 안)';
  ok(`${name} — CLIP이 도는 동안 YAMNet은 기다리지 않는다(줄이 따로)`, async () => {
    const d = offscreenDoc({ worker: mode, clipMs: 200 });
    await d.send({ to: 'sb-offscreen', op: 'status' });
    await d.send({ to: 'sb-offscreen', op: 'clipStatus' });
    const t0 = Date.now();
    const c = d.send({ to: 'sb-offscreen', op: 'clip', rgb: d.rgb() });
    await sleep(20);                                    // CLIP이 돌기 시작한 뒤
    const y = await d.send({ to: 'sb-offscreen', op: 'yamnet', pcm: d.pcm() });
    const yMs = Date.now() - t0;
    assert.ok(y.ok, y.error);
    assert.ok(yMs < 150, `YAMNet이 ${yMs} ms 걸렸다 — CLIP(200 ms) 뒤에서 기다렸다`);
    assert.ok((await c).ok);
    assert.deepStrictEqual(d.log.filter((x) => x !== 'clip-end').slice(0, 2), ['clip-start', 'yamnet']);
  });
}

ok('Worker — CLIP이 Worker로 가고, 프레임은 옮겨지며(transfer), 점수는 예전 길과 같다', async () => {
  const w = offscreenDoc({ worker: 'ok' });
  const st = (await w.send({ to: 'sb-offscreen', op: 'clipStatus' })).result;
  assert.strictEqual(st.mode, 'worker');
  assert.strictEqual(st.probe, 'blood');
  const rw = (await w.send({ to: 'sb-offscreen', op: 'clip', rgb: w.rgb() })).result;
  assert.strictEqual(w.transfers.length, 1, 'rgb 버퍼를 옮겨야 한다');
  const n = offscreenDoc({ worker: 'none' });
  const rn = (await n.send({ to: 'sb-offscreen', op: 'clip', rgb: n.rgb() })).result;
  for (const k of ['blood', 'syringe', 'spider']) assert.ok(Math.abs(rw.zs[k] - rn.zs[k]) < 1e-12, k);
  assert.ok(Math.abs(rw.probe.blood - rn.probe.blood) < 1e-12);
});

ok('Worker를 못 띄우면 이 문서 안에서 돈다(mode inline)', async () => {
  const d = offscreenDoc({ worker: 'fail' });
  const st = (await d.send({ to: 'sb-offscreen', op: 'clipStatus' })).result;
  assert.ok(st.ready, JSON.stringify(st));
  assert.strictEqual(st.mode, 'inline');
  assert.ok((await d.send({ to: 'sb-offscreen', op: 'clip', rgb: d.rgb() })).ok);
});

ok('모델이 없으면 Worker에서도 물러나지 않고 「CLIP 모델 없음」', async () => {
  const d = offscreenDoc({ worker: 'ok', headsOk: false });
  const st = (await d.send({ to: 'sb-offscreen', op: 'clipStatus' })).result;
  assert.strictEqual(st.ready, false);
  assert.match(st.error, /CLIP 모델 없음/);
  const r = await d.send({ to: 'sb-offscreen', op: 'clip', rgb: d.rgb() });
  assert.strictEqual(r.ok, false);
  assert.match(r.error, /CLIP 모델 없음/);
});

ok('상태 확인은 줄에 서지 않는다 — CLIP이 도는 중에도 status가 바로 온다', async () => {
  const d = offscreenDoc({ worker: 'none', clipMs: 200 });
  await d.send({ to: 'sb-offscreen', op: 'clipStatus' });
  const c = d.send({ to: 'sb-offscreen', op: 'clip', rgb: d.rgb() });
  await sleep(20);
  const t0 = Date.now();
  const st = await d.send({ to: 'sb-offscreen', op: 'clipStatus' });
  assert.ok(Date.now() - t0 < 100, 'clipStatus가 CLIP 뒤에서 기다렸다');
  assert.ok(st.result.ready);
  await c;
});

(async () => {
  console.log('offscreen 줄 나누기 · CLIP Worker');
  for (const [name, fn] of tests) {
    total++;
    try { await fn(); console.log(`  ✓ ${name}`); pass++; }
    catch (e) { console.error(`  ✗ ${name}\n    ${e.stack.split('\n').slice(0, 3).join('\n    ')}`); process.exitCode = 1; }
  }
  console.log(`${pass}/${total} 통과`);
})();
