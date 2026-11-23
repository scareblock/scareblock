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
    style: {}, textContent: '', parentNode: null, ls: {},
    addEventListener(t, f) { (this.ls[t] ||= []).push(f); },
    remove() { if (this.parentNode) { this.parentNode.children = this.parentNode.children.filter((c) => c !== this); this.parentNode = null; } },
  });
  const host = { children: [], appendChild(c) { this.children.push(c); c.parentNode = this; },
    contains(n) { return n === this || this.children.includes(n) || n?.inPlayer; } };
  const head = { children: [], appendChild(c) { this.children.push(c); c.parentNode = this; } };
  return {
    host,
    doc: { createElement: mk, head },
    win: {
      addEventListener: (t, f) => keyLs.push({ t, f }),
      removeEventListener: (t, f) => { const i = keyLs.findIndex((x) => x.t === t && x.f === f); if (i >= 0) keyLs.splice(i, 1); },
    },
    /** 실제 키 입력처럼 keydown → keyup 둘 다 보낸다. 둘 중 하나라도 막혔으면 stopped */
    press(key, types = ['keydown', 'keyup']) {
      const res = { stopped: false, prevented: false };
      for (const type of types) {
        const e = { type, key, code: key === ' ' ? 'Space' : key,
          preventDefault() { res.prevented = true; }, stopImmediatePropagation() { res.stopped = true; } };
        for (const { t, f } of [...keyLs]) if (t === type) f(e);
      }
      return res;
    },
    keyListeners() { return keyLs.length; },
    /** 플레이어 안을 window 캡처 단계에서 클릭 — 유튜브 캡처 리스너보다 먼저 끊기는지 */
    capClick(inPlayer = true) {
      const res = { stopped: false };
      const e = { type: 'click', target: { inPlayer }, preventDefault() {}, stopImmediatePropagation() { res.stopped = true; } };
      for (const { t, f } of [...keyLs]) if (t === 'click') f(e);
      return res;
    },
    click() {   // 덮개가 클릭을 받아 버리는지 — 버리지 않으면 유튜브가 재생 · 정지를 바꾼다
      const ov = host.children[0];
      let stopped = false;
      const e = { preventDefault() {}, stopPropagation() { stopped = true; } };
      for (const f of ov?.ls?.click || []) f(e);
      return { covered: !!ov, stopped };
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
    assert.strictEqual(dom.overlayText(), '');                         // 글자는 없고
    assert.strictEqual(dom.host.children[0].style.background, 'transparent');   // 투명한 덮개만 남는다
    v.run(299);
    assert.strictEqual(s.state, 'running');
    v.run(2);
    assert.strictEqual(s.state, 'done');
    assert.ok(v.paused);
    const r = s.report();
    assert.ok(r.유효, JSON.stringify(r));
    // 끝은 절대 시각(시작점 + 300)이라 두 조건이 같은 구간에서 끝난다. 시작은 멈추는 순간 한 프레임까지 넘을 수 있다
    assert.ok(r.끝_보이는시각 >= 400 && r.끝_보이는시각 < 400 + 1.5 / FPS, JSON.stringify(r));   // 판정한 프레임 시각
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

ok('Space를 뗄 때(keyup)도 막는다 — 유튜브가 keyup으로 다시 멈추던 것 (11.23 실기)', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(3, v, dom);
  s.prepare({ start: 50, dur: 20 });
  await tick();
  v.run(5);
  const r = dom.press(' ');                        // keydown → 시작, keyup → 막혀야 한다
  assert.ok(r.stopped);
  assert.strictEqual(s.state, 'running');
  assert.ok(!v.paused);
  assert.ok(dom.press(' ', ['keyup']).stopped, '진행 중 keyup도 막는다');
});

ok('진행 중 숫자 · k · 화살표 · m은 막고 f(전체 화면)는 둔다', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(0, v, dom);
  s.prepare({ start: 50, dur: 20 });
  await tick();
  dom.press(' ');
  for (const k of ['5', 'k', 'ArrowLeft', 'm', ' ']) assert.ok(dom.press(k).stopped, k);
  assert.strictEqual(dom.press('f').stopped, false);
  assert.strictEqual(s.state, 'running');
});

ok('준비됨 · 진행 중에는 덮개가 클릭을 받아 버린다', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(3, v, dom);
  s.prepare({ start: 50, dur: 20 });
  await tick();
  v.run(5);
  let c = dom.click();
  assert.ok(c.covered && c.stopped, '준비됨');
  dom.press(' ');
  c = dom.click();
  assert.ok(c.covered && c.stopped, '진행 중 — 화면은 비었어도 덮개는 있다');
  assert.strictEqual(dom.host.children[0].style.background, 'transparent');
});

ok('플레이어 안 클릭은 window 캡처에서 끊고, 밖(DevTools · 페이지 다른 곳)은 그대로 둔다', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(0, v, dom);
  s.prepare({ start: 50, dur: 20 });
  await tick();
  assert.ok(dom.capClick(true).stopped, '준비됨');
  assert.strictEqual(dom.capClick(false).stopped, false);
  dom.press(' ');
  assert.ok(dom.capClick(true).stopped, '진행 중');
  v.run(21);
  assert.strictEqual(dom.capClick(true).stopped, false, '끝나면 돌려준다');
});

ok('Esc — 참가자 중단은 기록되고 무효, 화면에 「중단했습니다」', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(3, v, dom);
  s.prepare({ start: 50, dur: 20 });
  await tick();
  v.run(5);
  dom.press(' ');
  v.run(3);
  dom.press('Escape', ['keydown']);
  assert.strictEqual(s.state, 'stopped');
  assert.ok(v.paused);
  const r = s.report();
  assert.strictEqual(r.유효, false);
  assert.ok(r.끼어든일.some((x) => x.includes('참가자 중단')), JSON.stringify(r));
  assert.strictEqual(dom.overlayText(), '중단했습니다');
});

ok('끝나면 키를 돌려준다', async () => {
  const v = fakeVideo(0), dom = fakeDom(), s = session(0, v, dom);
  s.prepare({ start: 50, dur: 5 });
  await tick();
  dom.press(' ');
  v.run(6);
  assert.strictEqual(s.state, 'done');
  assert.strictEqual(dom.keyListeners(), 0);
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
