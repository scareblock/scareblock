/**
 * offscreen 문서 — YAMNet을 onnxruntime-web(wasm)으로 돌린다. 확장 출처라 유튜브 CSP를 받지 않는다.
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
  let queue = Promise.resolve();   // wasm 세션은 한 번에 하나씩 돌린다

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

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.to !== 'sb-offscreen') return false;
    const run = msg.op === 'yamnet' ? () => yamnet(msg.pcm)
      : msg.op === 'status' ? async () => {
        await load().catch(() => {});
        return { ...state, classes: state.classes ? state.classes.length : null, names: state.classes };
      }
        : null;
    if (!run) return false;
    queue = queue.then(run).then(
      (result) => sendResponse({ ok: true, result }),
      (e) => sendResponse({ ok: false, error: String(e?.message || e) }),
    );
    return true;
  });
})();
