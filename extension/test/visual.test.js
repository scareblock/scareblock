/**
 * 시각 특징 검증.
 *   node extension/test/visual.test.js
 *
 * 합성 휘도 프레임으로 세 특징이 각자 맞는 것에만 반응하는지 본다 —
 * 페이드는 모션이 아니고, 패닝은 컷이 아니고, 장면 교체는 컷이다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/visual.js'), 'utf8'));
const { VisualCore } = window.SB;

let pass = 0;
const ok = (name, fn) => {
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
};

const W = 64, H = 36;
const frame = (f) => {
  const y = new Uint8Array(W * H);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) y[r * W + c] = f(c, r);
  return y;
};
// 가로 그라디언트 — 패닝하면 내용이 옆으로 밀린다
const grad = (dx = 0, base = 40) => frame((c) => base + ((c + dx) % W) * 2);

console.log('시각 특징');

ok('첫 프레임은 차이 특징이 null', () => {
  const v = new VisualCore();
  const f = v.push(grad(), 0);
  assert.strictEqual(f.dMean, null);
  assert.strictEqual(f.hist, null);
  assert.strictEqual(f.motion, null);
  assert.strictEqual(f.cut, false);
  assert.strictEqual(f.flash, false);
});

ok('같은 프레임이면 전부 0', () => {
  const v = new VisualCore();
  v.push(grad(), 0);
  const f = v.push(grad(), 0.5);
  assert.strictEqual(f.dMean, 0);
  assert.strictEqual(f.hist, 0);
  assert.strictEqual(f.motion, 0);
});

ok('장면 교체는 컷이다', () => {
  const v = new VisualCore();
  v.push(frame(() => 30), 0);                       // 어두운 단색
  const f = v.push(frame((c, r) => ((c + r) % 2 ? 230 : 120)), 0.5);  // 밝은 체커
  assert.ok(f.cut, `hist=${f.hist}`);
});

ok('페이드(전역 밝기 이동)는 모션이 아니다', () => {
  const v = new VisualCore();
  v.push(grad(0, 40), 0);
  const f = v.push(grad(0, 100), 0.5);              // 모든 픽셀 +60
  assert.ok(Math.abs(f.dMean - 60 / 255) < 1e-6);
  assert.ok(f.flash, `dMean=${f.dMean}`);
  assert.ok(f.motion < 1e-6, `motion=${f.motion}`);
});

ok('패닝은 모션이지 컷이 아니다', () => {
  const v = new VisualCore();
  v.push(grad(0), 0);
  const f = v.push(grad(8), 0.5);
  assert.ok(f.motion > 0.02, `motion=${f.motion}`);
  assert.ok(!f.cut, `hist=${f.hist}`);              // 같은 값들이 자리만 바뀌었다
  assert.ok(!f.flash, `dMean=${f.dMean}`);
});

ok('작은 밝기 변화는 밝기 급변이 아니다', () => {
  const v = new VisualCore();
  v.push(grad(0, 40), 0);
  const f = v.push(grad(0, 50), 0.5);
  assert.ok(!f.flash, `dMean=${f.dMean}`);
});

ok('시각이 뒤로 가면(시크) 차이를 내지 않는다', () => {
  const v = new VisualCore();
  v.push(frame(() => 30), 10);
  const f = v.push(frame(() => 230), 4, 1.25);
  assert.strictEqual(f.dMean, null);
  assert.strictEqual(f.cut, false);
});

ok('간격이 maxGap을 넘으면(탭 숨김) 차이를 내지 않는다', () => {
  const v = new VisualCore();
  v.push(frame(() => 30), 0);
  const f = v.push(frame(() => 230), 5, 1.25);
  assert.strictEqual(f.dMean, null);
  const g = v.push(frame(() => 230), 5.5, 1.25);   // 그 다음부터는 다시 잰다
  assert.strictEqual(g.dMean, 0);
});

ok('휘도 변환 — 흰색 255, 검정 0, 순수 초록이 가장 밝다', () => {
  const px = (r, g, b) => VisualCore.luma(new Uint8ClampedArray([r, g, b, 255]), new Uint8Array(1))[0];
  assert.strictEqual(px(255, 255, 255), 255);
  assert.strictEqual(px(0, 0, 0), 0);
  assert.ok(px(0, 255, 0) > px(255, 0, 0) && px(255, 0, 0) > px(0, 0, 255));
});

console.log(`${pass}/9 통과`);
