/**
 * offscreen 문서 — YAMNet과 CLIP을 onnxruntime-web(wasm)으로 돌린다. 확장 출처라 유튜브 CSP를 받지 않는다.
 *
 * 모델은 `docs/yamnet-setup.md`로 만든 것과 같은 파일이다(`eval/`이 쓰는 것). 입력은 16 kHz 모노 파형,
 * 출력 `scores`는 [패치 수, 521]. 0.975초를 넣으면 패치가 하나라 첫 줄만 돌려준다.
 *
 * 세션은 첫 요청 때 만든다. 모델이나 런타임이 없으면 status가 그 이유를 돌려주고, 추론 요청은 실패로 답한다 —
 * 페이지 쪽은 그때 null을 받아 지금처럼(검증 없이) 돈다.
 */
(() => {
  const MODEL = 'models/yamnet.onnx';
  const CLASSMAP = 'models/yamnet_class_map.csv';
  const state = { ready: false, error: null, loadMs: null, runs: 0, runMs: [], classes: null };
  let session = null;
  let loading = null;
  // 줄은 모델마다 따로 — 같은 세션은 한 번에 하나씩, 다른 모델은 서로 기다리지 않는다(#54 리뷰 🟡2).
  // 상태 확인(status · clipStatus)은 줄에 서지 않는다 — 모델을 올리는 약속(promise)만 같이 쓴다
  let yamnetQueue = Promise.resolve();
  let clipQueue = Promise.resolve();

  function load() {
    if (!loading) {
      loading = (async () => {
        if (typeof ort === 'undefined') {
          throw new Error('onnxruntime-web이 없다 — extension/fetch-deps.sh를 돌린다');
        }
        const t0 = performance.now();
        ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/');
        ort.env.wasm.numThreads = 1;   // 스레드는 cross-origin isolation이 필요하다. 패치 하나라 한 스레드로 충분하다
        session = await ort.InferenceSession.create(chrome.runtime.getURL(MODEL), {
          executionProviders: ['wasm'],
        });
        if (!session.outputNames.includes('scores')) {
          throw new Error(`모델 출력에 scores가 없다: ${session.outputNames.join(', ')}`);
        }
        state.classes = await loadClasses();
        state.loadMs = Math.round(performance.now() - t0);
        state.ready = true;
        state.error = null;
      })().catch((e) => {
        state.error = String(e?.message || e);
        loading = null;            // 파일을 채운 뒤 다시 시도할 수 있게
        throw e;
      });
    }
    return loading;
  }

  async function loadClasses() {
    try {
      const txt = await (await fetch(chrome.runtime.getURL(CLASSMAP))).text();
      // index,mid,display_name — display_name에 쉼표가 있으면 따옴표로 싸여 있다
      return txt.trim().split('\n').slice(1).map((l) => {
        const m = l.match(/^(\d+),([^,]*),(.*)$/);
        return m ? m[3].replace(/^"(.*)"$/, '$1') : l;
      });
    } catch {
      return null;    // 이름은 디버그용이다 — 없어도 점수는 낸다
    }
  }

  async function yamnet(b64) {
    await load();
    const pcm = SB.pcm.decode(b64);
    if (pcm.length !== SB.YAMNET_WIN_N) throw new Error(`표본 ${pcm.length}개 — ${SB.YAMNET_WIN_N}개여야 한다`);
    const t0 = performance.now();
    const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', pcm, [pcm.length]) };
    const out = await session.run(feeds, ['scores']);
    const sc = out.scores;
    const ms = performance.now() - t0;
    state.runs++;
    state.runMs.push(Math.round(ms * 10) / 10);
    if (state.runMs.length > 50) state.runMs.shift();
    const n = sc.dims[sc.dims.length - 1];
    return { scores: Array.from(sc.data.subarray(0, n)), patches: sc.dims[0], ms };
  }

  // ── CLIP ViT-B/32 (이미지 쪽) ─────────────────────────────────────────
  // 모델은 eval/clip_zeroshot이 쓰는 _local/models/clip/visual.onnx와 같은 파일(fp32)이다 — 줄이면 점수가 달라져
  // 오프라인에서 정한 θ가 맞지 않는다. heads는 문장 임베딩 · logit_scale · 피 보강 모델 계수(fetch-deps.sh가 만든다).
  // YAMNet과 따로 올린다 — 하나가 없어도 다른 하나는 돈다.
  //
  // **CLIP은 전용 Worker(clip-worker.js)에서 돈다.** 이 문서에서 돌리면 CLIP 한 장(약 210 ms) 동안 YAMNet이 같은 스레드에서
  // 기다린다(CLIP 켜면 YAMNet 왕복 26 → 296 ms, #64 실기). Worker를 못 띄우면 예전처럼 이 문서 안에서 돈다(mode: 'inline').
  const CLIP_MODEL = 'models/clip_visual.onnx';
  const CLIP_HEADS = 'models/clip_heads.json';
  const CLIP_MISSING = `CLIP 모델 없음 — docs/clip-setup.md대로 만든 뒤 extension/fetch-deps.sh (${CLIP_HEADS})`;
  const clipState = { ready: false, error: null, loadMs: null, runs: 0, runMs: [], probe: null, mode: null };
  let clipSession = null;
  let clipHeads = null;
  let clipLoading = null;
  let clipWorker = null;   // { w, seq, pending: Map }

  /** 전용 Worker를 띄우고 모델을 올린다. Worker를 못 띄우면 던진다 — 부르는 쪽이 이 문서 안으로 물러난다 */
  function loadInWorker() {
    return new Promise((resolve, reject) => {
      let w;
      try { w = new Worker(chrome.runtime.getURL('src/clip-worker.js')); } catch (e) { reject(e); return; }
      const cw = { w, seq: 0, pending: new Map() };
      w.onerror = (e) => {
        e.preventDefault?.();
        const err = new Error(`CLIP Worker 시작 실패${e.message ? ` — ${e.message}` : ''}`);
        for (const p of cw.pending.values()) p.reject(err);
        cw.pending.clear();
        if (clipWorker === cw) { clipWorker = null; clipLoading = null; clipState.ready = false; clipState.error = err.message; }
        reject(err);
      };
      w.onmessage = (e) => {
        const m = e.data || {};
        if (m.op === 'load') {
          if (!m.ok) { w.terminate(); reject(new Error(m.error)); return; }
          clipWorker = cw;
          resolve(m);
        } else if (m.op === 'run') {
          const p = cw.pending.get(m.id);
          if (!p) return;
          cw.pending.delete(m.id);
          m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
        }
      };
      w.postMessage({ op: 'load', modelUrl: chrome.runtime.getURL(CLIP_MODEL),
        headsUrl: chrome.runtime.getURL(CLIP_HEADS), wasmPaths: chrome.runtime.getURL('vendor/') });
    });
  }

  /** 예전 방식 — 이 문서 안에서 올린다 */
  async function loadInline() {
    if (typeof ort === 'undefined') throw new Error('onnxruntime-web이 없다 — extension/fetch-deps.sh를 돌린다');
    if (!SB.clipNormalize) throw new Error('clip-prep.js가 안 올라왔다');
    ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/');
    ort.env.wasm.numThreads = 1;
    // 확장 안에 없는 파일을 fetch하면 404가 아니라 「Failed to fetch」로 던진다 — 원인을 알 수 있게 바꿔 말한다
    const r = await fetch(chrome.runtime.getURL(CLIP_HEADS)).catch(() => null);
    if (!r?.ok) throw new Error(CLIP_MISSING);
    clipHeads = await r.json();
    clipSession = await ort.InferenceSession.create(chrome.runtime.getURL(CLIP_MODEL), {
      executionProviders: ['wasm'],
    });
    return { probe: clipHeads.probe ? clipHeads.probe.cat : null };
  }

  function loadClip() {
    if (!clipLoading) {
      clipLoading = (async () => {
        const t0 = performance.now();
        let r = null;
        if (typeof Worker !== 'undefined') {
          try {
            r = await loadInWorker();
            clipState.mode = 'worker';
          } catch (e) {
            if (String(e?.message).startsWith('CLIP 모델 없음')) throw e;   // 파일이 없으면 이 문서에서도 못 올린다
            console.warn('[scareblock] CLIP Worker를 못 띄웠다 — offscreen 안에서 돌린다(YAMNet이 CLIP 뒤에서 기다릴 수 있다)', e);
          }
        }
        if (!r) { r = await loadInline(); clipState.mode = 'inline'; }
        clipState.probe = r.probe;
        clipState.loadMs = Math.round(performance.now() - t0);
        clipState.ready = true;
        clipState.error = null;
      })().catch((e) => {
        clipState.error = String(e?.message || e);
        clipLoading = null;
        throw e;
      });
    }
    return clipLoading;
  }

  async function clip(b64) {
    await loadClip();
    const rgb = SB.pcm.decodeBytes(b64);
    let result;
    if (clipState.mode === 'worker' && clipWorker) {
      const cw = clipWorker;
      const id = ++cw.seq;
      result = await new Promise((resolve, reject) => {
        cw.pending.set(id, { resolve, reject });
        cw.w.postMessage({ op: 'run', id, rgb }, [rgb.buffer]);
      });
    } else {
      const x = SB.clipNormalize(rgb);
      const t0 = performance.now();
      const out = await clipSession.run({ [clipSession.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, 224, 224]) });
      result = { ...SB.clipHeads(out[clipSession.outputNames[0]].data, clipHeads), ms: performance.now() - t0 };
    }
    clipState.runs++;
    clipState.runMs.push(Math.round(result.ms));
    if (clipState.runMs.length > 50) clipState.runMs.shift();
    return result;
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.to !== 'sb-offscreen') return false;
    const run = msg.op === 'yamnet' ? () => yamnet(msg.pcm)
      : msg.op === 'clip' ? () => clip(msg.rgb)
        : msg.op === 'status' ? async () => {
          await load().catch(() => {});
          return { ...state, classes: state.classes ? state.classes.length : null, names: state.classes };
        }
          : msg.op === 'clipStatus' ? async () => {
            await loadClip().catch(() => {});
            return { ...clipState };
          }
            : null;
    if (!run) return false;
    const done = [(result) => sendResponse({ ok: true, result }), (e) => sendResponse({ ok: false, error: String(e?.message || e) })];
    if (msg.op === 'yamnet') yamnetQueue = yamnetQueue.then(run).then(...done);
    else if (msg.op === 'clip') clipQueue = clipQueue.then(run).then(...done);
    else run().then(...done);   // 상태 확인은 줄에 서지 않는다
    return true;
  });
})();
