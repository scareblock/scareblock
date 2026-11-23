/**
 * E5 진행 도우미 검증 — 가짜 <video>로 시작 · 끝이 두 조건에서 같은 「보이는 시각」에 맞는지 본다.
 *   node extension/test/e5-session.test.js
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

eval(fs.readFileSync(path.join(__dirname, '../tools/e5-session.js'), 'utf8'));
const { E5Session } = globalThis;

let pass = 0, total = 0;
const tests = [];
const ok = (name, fn) => tests.push([name, fn]);

const FPS = 30;

/** 재생 중이면 step()마다 1/FPS초씩 가고, 프레임 콜백을 부른다 */
function fakeVideo(t = 0) {
  const ls = {};
  const v = {
    _t: t, paused: true, playbackRate: 1, cbs: [],
    get currentTime() { return this._t; },
    set currentTime(x) {
      this._t = x;
      fire('seeking');
      Promise.resolve().then(() => fire('seeked'));
    },
    play() { this.paused = false; fire('play'); return Promise.resolve(); },
    pause() { if (!this.paused) { this.paused = true; fire('pause'); } },
    addEventListener(e, f) { (ls[e] ||= []).push(f); },
    removeEventListener(e, f) { ls[e] = (ls[e] || []).filter((x) => x !== f); },
    requestVideoFrameCallback(f) { this.cbs.push(f); },
    step() {
      if (this.paused) return;
      this._t += 1 / FPS;
      const cbs = this.cbs; this.cbs = [];
      for (const f of cbs) f(0, { mediaTime: this._t });
    },
    run(sec) { for (let i = 0; i < Math.round(sec * FPS); i++) this.step(); },
  };
  function fire(e) { for (const f of [...(ls[e] || [])]) f(); }
  return v;
}

function fakeDom() {
  const keyLs = [];
  const mk = () => ({
    style: {}, textContent: '', parentNode: null,
    remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter((c) => c !== this); this.parentNode = null; } },
  });
  const host = { children: [], appendChild(c) { this.children.push(c); c.parentNode = this; } };
  const head = { children: [], appendChild(c) { this.children.push(c); c.parentNode = this; } };
  return {
    host,
    doc: { createElement: mk, head },
    win: {
      addEventListener: (t, f) => keyLs.push(f),
      removeEventListener: (t, f) => { const i = keyLs.indexOf(f); if (i >= 0) keyLs.splice(i, 1); },
    },
    press(key) {
      const e = { key, code: key === ' ' ? 'Space' : key, stopped: false, prevented: false,
        preventDefault() { this.prevented = true; }, stopImmediatePropagation() { this.stopped = true; } };
      for (const f of [...keyLs]) f(e);
      return e;
    },
    overlayText() { return host.children[0]?.textContent ?? null; },
  };
}

function session(delay, video, dom) {
  let wall = 0;
  return new E5Session({
    video, host: dom.host, doc: dom.doc, win: dom.win, delay, videoId: 'abc',
    now: () => (wall += 0), log: () => {},
  });
}
const tick = () => new Promise((r) => setImmediate(r));

for (const delay of [0, 3]) {
  const cond = delay ? '필터' : '원본';

  ok(`${cond} — 준비되면 화면에 시작 프레임이 보이는 자리에 멈춰 있다`, async () => {
    const v = fakeVideo(10), dom = fakeDom(), s = session(delay, v, dom);
    s.prepare({ start: 100, video: 'abc' });
    assert.strictEqual(dom.overlayText(), '준비 중…');
    await tick();
    v.run(delay + 2);                                    // 버퍼 채우기 (필터만 실제로 돈다)
    assert.strictEqual(s.state, 'ready');
    assert.ok(v.paused);
    const shown = v.currentTime - delay;
    assert.ok(shown >= 100 && shown < 100 + 1.5 / FPS, `보이는 시각 ${shown}`);
    assert.strictEqual(dom.overlayText(), '준비됨 — Space 키로 시작');
  });

  ok(`${cond} — Space로 시작해 300초가 보이면 멈추고 「끝」`, async () => {
    const v = fakeVideo(10), dom = fakeDom(), s = session(delay, v, dom);
    s.prepare({ start: 100 });
    await tick();
    v.run(delay + 2);
    const e = dom.press(' ');
    assert.ok(e.stopped && e.prevented, '유튜브가 Space를 먼저 받으면 재생 · 정지가 두 번 일어난다');
    assert.strictEqual(s.state, 'running');
    assert.strictEqual(dom.overlayText(), null);
    v.run(299);
    assert.strictEqual(s.state, 'running');
    v.run(2);
    assert.strictEqual(s.state, 'done');
    assert.ok(v.paused);
    const r = s.report();
    assert.ok(r.유효, JSON.stringify(r));
    // 끝은 절대 시각(시작점 + 300)이라 두 조건이 같은 구간에서 끝난다. 시작은 멈추는 순간 한 프레임까지 넘을 수 있다
    assert.ok(r.끝_보이는시각 >= 400 && r.끝_보이는시각 < 400 + 2 / FPS, JSON.stringify(r));
    assert.ok(Math.abs(r.끝_보이는시각 - r.시작_보이는시각 - 300) < 2 / FPS, JSON.stringify(r));
    assert.strictEqual(dom.overlayText(), '끝');
  });
}

ok('두 조건이 같은 보이는 구간을 보여준다', async () => {
  const spans = [];
  for (const delay of [0, 3]) {
    const v = fakeVideo(0), dom = fakeDom(), s = session(delay, v, dom);
    s.prepare({ start: 50 });
    await tick();
    v.run(delay + 2);
    dom.press(' ');
    v.run(305);
    const r = s.report();
    spans.push([r.시작_보이는시각, r.끝_보이는시각]);
  }
  assert.ok(Math.abs(spans[0][0] - spans[1][0]) < 2 / FPS && Math.abs(spans[0][1] - spans[1][1]) < 2 / FPS,
    JSON.stringify(spans));
});

ok('진행 중 일시정지 · 시크는 기록되고 그 세션은 유효하지 않다', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(3, v, dom);
  s.prepare({ start: 50 });
  await tick();
  v.run(5);
  dom.press(' ');
  v.run(10);
  v.pause();
  v.play();
  v.currentTime = v.currentTime + 5;
  await tick();
  v.run(300);
  const r = s.report();
  assert.strictEqual(r.유효, false);
  assert.strictEqual(r.끼어든일.length, 2, JSON.stringify(r.끼어든일));
});

ok('준비 전 Space · 준비 중 유튜브 단축키는 막지 않는다 / 준비됨에서는 k · 화살표를 막는다', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(0, v, dom);
  assert.strictEqual(dom.press(' ').stopped, false);   // 아직 리스너 없음
  s.prepare({ start: 20 });
  await tick();
  assert.strictEqual(s.state, 'ready');
  assert.strictEqual(dom.press('ArrowRight').stopped, true);
  assert.strictEqual(s.state, 'ready');
});

ok('시작점이 비어 있으면 무엇을 채워야 하는지 알려준다', () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(0, v, dom);
  assert.throws(() => s.prepare('c019'), /clips-e1\.csv/);
});

ok('다른 영상 페이지면 시작하지 않는다', () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(0, v, dom);
  assert.throws(() => s.prepare({ start: 1, video: 'zzz' }), /가 아니다/);
});

ok('abort — 화면 · 키 · 감시를 원래대로', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(3, v, dom);
  s.prepare({ start: 50 });
  await tick();
  v.run(5);
  s.abort();
  assert.strictEqual(dom.overlayText(), null);
  assert.strictEqual(dom.doc.head.children.length, 0);
  assert.strictEqual(dom.press(' ').stopped, false);
  assert.strictEqual(s.state, 'idle');
});

(async () => {
  console.log('E5 진행 도우미');
  for (const [name, fn] of tests) {
    total++;
    try { await fn(); console.log(`  ✓ ${name}`); pass++; }
    catch (e) { console.error(`  ✗ ${name}\n    ${e.stack.split('\n').slice(0, 3).join('\n    ')}`); process.exitCode = 1; }
  }
  console.log(`${pass}/${total} 통과`);
})();
