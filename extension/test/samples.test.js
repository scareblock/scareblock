/**
 * 표본 수집기 검증.
 *   node extension/test/samples.test.js
 *
 * 백분위수가 틀리면 --render-s 가 틀리고, 그 값이 E1 적시성에 그대로 들어간다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/samples.js'), 'utf8'));
const { Samples } = window.SB;

let pass = 0;
const ok = (name, fn) => {
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
};

const fill = (s, arr) => { arr.forEach((v) => s.push(v)); return s; };

console.log('표본 수집기');

ok('빈 수집기는 null', () => {
  const s = new Samples();
  assert.strictEqual(s.count, 0);
  assert.strictEqual(s.percentile(90), null);
  assert.strictEqual(s.summary(), null);
});

ok('1~100에서 백분위수', () => {
  const s = fill(new Samples(), Array.from({ length: 100 }, (_, i) => i + 1));
  assert.strictEqual(s.percentile(50), 50);
  assert.strictEqual(s.percentile(90), 90);
  assert.strictEqual(s.percentile(100), 100);
});

ok('넣은 순서와 무관하다', () => {
  const a = fill(new Samples(), [5, 1, 4, 2, 3]);
  const b = fill(new Samples(), [1, 2, 3, 4, 5]);
  assert.strictEqual(a.percentile(90), b.percentile(90));
});

ok('용량을 넘기면 최근 것만 남는다', () => {
  const s = new Samples(4);
  fill(s, [100, 100, 100, 100, 1, 2, 3, 4]);
  assert.strictEqual(s.count, 4);
  assert.strictEqual(s.percentile(100), 4, '옛 100이 남아 있으면 안 된다');
});

ok('한 개만 있으면 모든 백분위수가 그 값', () => {
  const s = fill(new Samples(), [7.5]);
  assert.strictEqual(s.percentile(50), 7.5);
  assert.strictEqual(s.percentile(90), 7.5);
});

ok('summary 는 소수 둘째 자리', () => {
  const s = fill(new Samples(), [1.234, 2.345, 3.456]);
  const r = s.summary();
  assert.strictEqual(r.n, 3);
  assert.strictEqual(r.max, 3.46);
});

ok('정렬이 사전순이 아니라 수치순이다', () => {
  // Float64Array.sort 는 수치순이지만, 일반 배열이었다면 [1,10,2] 가 된다
  const s = fill(new Samples(), [10, 2, 1]);
  assert.strictEqual(s.percentile(100), 10);
  assert.strictEqual(s.percentile(1), 1);
});

ok('reset 뒤에는 비어 있다', () => {
  const s = fill(new Samples(), [1, 2, 3]);
  s.reset();
  assert.strictEqual(s.count, 0);
  assert.strictEqual(s.summary(), null);
});

ok('summary 는 percentile 과 같은 값을 낸다 (한 번 정렬)', () => {
  const s = fill(new Samples(8), [9, 3, 7, 1, 5, 2, 8, 4, 6, 10]);  // 용량 초과 포함
  const r = s.summary();
  assert.strictEqual(r.p50, s.percentile(50));
  assert.strictEqual(r.p90, s.percentile(90));
  assert.strictEqual(r.max, s.percentile(100));
});

console.log(`\n${pass}/9 통과`);
