/**
 * CLIP 전용 Worker — offscreen 문서 안에서 띄운다(확장 출처라 유튜브 CSP와 상관없다).
 *
 * **왜 따로 두나 (#54 리뷰 🟡2 · #64 실기).** offscreen 하나에서 YAMNet과 CLIP이 같은 스레드 · 같은 줄을 쓰면,
 * CLIP 한 장(약 210 ms, wasm 1스레드)이 도는 동안 YAMNet 요청이 기다린다 — CLIP을 켜면 YAMNet 왕복이 26 → 296 ms였다.
 * 말소리 기각(#52)은 시간이 모자라면 그냥 가리고, 사이렌이 붙으면 YAMNet 요청이 0.48초마다 생긴다.
 * CLIP을 여기로 옮기면 두 모델이 다른 스레드에서 돈다.
 *
 * 계산은 offscreen에서 하던 것 그대로다 — `clip-prep.js`의 clipNormalize · clipHeads, 같은 fp32 모델.
 *
 * 메시지: {op:'load', modelUrl, headsUrl, wasmPaths} → {op:'load', ok, loadMs, probe}
 *         {op:'run', id, rgb: Uint8Array(224·224·3)} → {op:'run', id, ok, result:{zs, probe, ms}}
 */
self.window = self;   // pcm.js · clip-prep.js는 window.SB에 붙는다
importScripts('../vendor/ort.wasm.min.js', 'pcm.js', 'clip-prep.js');

(() => {
  let session = null, heads = null;

  async function load({ modelUrl, headsUrl, wasmPaths }) {
    const t0 = performance.now();
    ort.env.wasm.wasmPaths = wasmPaths;
    ort.env.wasm.numThreads = 1;
    const r = await fetch(headsUrl).catch(() => null);
    if (!r?.ok) throw new Error(`CLIP 모델 없음 — docs/clip-setup.md대로 만든 뒤 extension/fetch-deps.sh (${headsUrl.split('/').pop()})`);
    heads = await r.json();
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    return { loadMs: Math.round(performance.now() - t0), probe: heads.probe ? heads.probe.cat : null };
  }

  async function run(rgb) {
    if (!session) throw new Error('CLIP이 아직 안 올라왔다');
    const x = SB.clipNormalize(rgb);
    const t0 = performance.now();
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, 224, 224]) });
    const emb = out[session.outputNames[0]].data;
    return { ...SB.clipHeads(emb, heads), ms: performance.now() - t0 };
  }

  self.onmessage = async (e) => {
    const m = e.data || {};
    try {
      if (m.op === 'load') self.postMessage({ op: 'load', ok: true, ...(await load(m)) });
      else if (m.op === 'run') self.postMessage({ op: 'run', id: m.id, ok: true, result: await run(m.rgb) });
    } catch (err) {
      self.postMessage({ op: m.op, id: m.id, ok: false, error: String(err?.message || err) });
    }
  };
})();
