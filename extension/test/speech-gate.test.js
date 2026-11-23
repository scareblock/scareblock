/**
 * 말소리 기각 검증. 브라우저 없이 돈다.
 *   node extension/test/speech-gate.test.js
 *
 * 규칙은 `eval/speech_gate.py`(#40 브랜치 `adb83cf` — main에는 아직 없다) gate()와 같아야 한다: Speech ≥ θ **이고** Speech > 20개 클래스 최댓값일 때만 버린다.
 * 판정을 못 하면(YAMNet 없음 · 패치 없음 · 시간 초과) **버리지 않고 낸다.**
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/config.js'), 'utf8'));
eval(fs.readFileSync(path.join(__dirname, '../src/speech-gate.js'), 'utf8'));
const { speechDecide, SpeechGate, SCARE_IDX } = window.SB;

let pass = 0;
const tests = [];
const ok = (name, fn) => tests.push([name, fn]);

const scores = (speech, scare = 0, at = 11) => {
  const s = new Float32Array(521);
  s[0] = speech;
  s[at] = scare;
  return s;
};
const trig = (time = 10) => ({ time, duration: 1.5, category: 'jumpscare', confidence: 1, source: 'rule' });
const gateWith = (yam, emitted) => new SpeechGate(() => yam, (t) => emitted.push(t));

console.log('말소리 기각');

ok('20개 클래스가 eval/yamnet.py CLASSES와 같다', () => {
  const py = fs.readFileSync(path.join(__dirname, '../../eval/yamnet.py'), 'utf8');
  const block = py.match(/CLASSES: dict\[int, str\] = \{([\s\S]*?)\}/)[1];
  const idx = [...block.matchAll(/(\d+):/g)].map((m) => +m[1]).sort((a, b) => a - b);
  assert.deepStrictEqual([...SCARE_IDX].sort((a, b) => a - b), idx);
  assert.strictEqual(idx.length, 20);
});

ok('조건 둘 다 맞으면 버린다', () => {
  assert.strictEqual(speechDecide(scores(0.5, 0.2)).reject, true);
});

ok('Speech가 θ 미만이면 버리지 않는다 (경계 0.1은 버린다)', () => {
  assert.strictEqual(speechDecide(scores(0.099, 0)).reject, false);
  assert.strictEqual(speechDecide(scores(0.1, 0)).reject, true);
});

ok('비명이 Speech보다 크면 버리지 않는다 — 조건 둘 다', () => {
  assert.strictEqual(speechDecide(scores(0.6, 0.7, 11)).reject, false);   // Screaming
  assert.strictEqual(speechDecide(scores(0.6, 0.6, 421)).reject, false);  // 같으면 「크지 않다」
});

ok('20개 밖의 클래스는 비교에 안 들어간다', () => {
  assert.strictEqual(speechDecide(scores(0.5, 0.9, 137)).reject, true);   // 137 = 음악
});

ok('통과하면 재생기에 넘기고, 기각하면 안 넘긴다', async () => {
  const out = [];
  const g1 = gateWith({ scoresAt: async () => scores(0.05) }, out);
  assert.strictEqual(await g1.submit(trig(), 2500), 'keep');
  const g2 = gateWith({ scoresAt: async () => scores(0.8) }, out);
  assert.strictEqual(await g2.submit(trig(), 2500), 'reject');
  assert.strictEqual(out.length, 1);
  assert.strictEqual(g2.stats.기각, 1);
});

ok('YAMNet이 없으면 판정 없이 낸다', async () => {
  const out = [];
  assert.strictEqual(await gateWith(null, out).submit(trig(), 2500), 'unseen');
  assert.strictEqual(out.length, 1);
});

ok('패치를 못 얻으면(null · 오류) 낸다', async () => {
  const out = [];
  assert.strictEqual(await gateWith({ scoresAt: async () => null }, out).submit(trig(), 2500), 'unseen');
  assert.strictEqual(await gateWith({ scoresAt: async () => { throw new Error('x'); } }, out).submit(trig(), 2500), 'unseen');
  assert.strictEqual(out.length, 2);
});

ok('표시 시각 전에 답이 없으면 낸다 — 늦은 답은 무시', async () => {
  const out = [];
  const slow = { scoresAt: () => new Promise((r) => setTimeout(() => r(scores(0.9)), 400)) };
  const g = gateWith(slow, out);
  assert.strictEqual(await g.submit(trig(), 300), 'timeout');   // 여유 250ms를 빼면 50ms
  assert.strictEqual(out.length, 1);
  await new Promise((r) => setTimeout(r, 450));
  assert.strictEqual(out.length, 1);   // 나중에 온 「말소리」 답이 이미 낸 것을 되돌리지 않는다
});

ok('남은 시간이 여유보다 짧으면 기다리지 않는다', async () => {
  const out = [];
  let asked = false;
  const g = gateWith({ scoresAt: async () => { asked = true; return scores(0.9); } }, out);
  assert.strictEqual(await g.submit(trig(), 100), 'timeout');
  assert.strictEqual(asked, false);
  assert.strictEqual(out.length, 1);
});

(async () => {
  for (const [name, fn] of tests) {
    try { await fn(); console.log(`  ✓ ${name}`); pass++; }
    catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
  }
  console.log(`\n${pass}/${tests.length} 통과`);
})();
