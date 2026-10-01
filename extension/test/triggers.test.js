/**
 * 트리거 조회 검증. 브라우저 없이 돈다.
 *   node extension/test/triggers.test.js
 *
 * triggerAt()은 매 렌더 프레임 불리므로 커서로 지난 트리거를 잘라낸다. 커서가
 * 잘못 전진하면 **걸려야 할 블러가 안 걸린다** — 눈으로는 "가끔 안 가려지네"
 * 정도로만 보여서 놓치기 쉽다. 그래서 선형 탐색과 대조한다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/config.js'), 'utf8'));
eval(fs.readFileSync(path.join(__dirname, '../src/samples.js'), 'utf8'));  // player 생성자가 쓴다
eval(fs.readFileSync(path.join(__dirname, '../src/player.js'), 'utf8'));
const { Player } = window.SB;

let pass = 0;
const ok = (name, fn) => {
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
};

// start()는 DOM을 쓰지만 constructor는 순수하다. 조회만 보므로 그대로 쓴다.
const make = () => new Player({}, window.SB.config);
const trig = (time, duration = 1.5) =>
  ({ time, duration, category: 'x', confidence: 1, source: 'test' });
const brute = (list, t) =>
  list.find((x) => t >= x.time && t <= x.time + x.duration) || null;

console.log('트리거 조회');

ok('구간 안이면 잡고 밖이면 null', () => {
  const p = make();
  p.addTriggers([trig(10, 1.5)]);
  assert.strictEqual(p.triggerAt(9.9), null);
  assert.ok(p.triggerAt(10));
  assert.ok(p.triggerAt(11.5));
  assert.strictEqual(p.triggerAt(11.6), null);
});

ok('계약에 맞지 않는 트리거는 버린다', () => {
  const p = make();
  const warn = console.warn;
  console.warn = () => {};
  p.addTriggers([{ time: 'x', duration: 1 }, { time: 1 }, trig(5)]);
  console.warn = warn;
  assert.strictEqual(p.triggers.length, 1);
});

ok('순서가 뒤섞여 들어와도 time 오름차순을 유지한다', () => {
  const p = make();
  p.addTriggers([trig(30), trig(10), trig(20), trig(5)]);
  const times = p.triggers.map((t) => t.time);
  assert.deepStrictEqual(times, [...times].sort((a, b) => a - b));
});

ok('커서가 전진한 뒤 앞쪽 트리거가 끼어도 찾는다', () => {
  const p = make();
  p.addTriggers([trig(10), trig(40)]);
  p.triggerAt(30);                       // 커서를 40 앞까지 밀어둔다
  p.addTriggers([trig(20)]);             // 지나간 자리에 늦게 도착
  assert.strictEqual(p.triggerAt(20.5)?.time, 20);
});

ok('500건 상한을 넘겨 앞이 잘려도 커서가 어긋나지 않는다', () => {
  const p = make();
  for (let i = 0; i < 520; i++) p.addTriggers([trig(i * 2)]);
  assert.strictEqual(p.triggers.length, 500);
  p.triggerAt(600);
  for (let i = 0; i < 520; i++) p.addTriggers([trig(1040 + i * 2)]);
  assert.ok(p._cursor <= p.triggers.length);
  assert.strictEqual(p.triggerAt(1500)?.time, 1500);
});

ok('커서를 0으로 되돌리면 과거 구간을 다시 찾는다 (시크·탭 복귀)', () => {
  const p = make();
  p.addTriggers([trig(10), trig(20), trig(30)]);
  p.triggerAt(31);
  assert.strictEqual(p.triggerAt(10.5), null, '커서가 전진했으면 과거는 안 보인다');
  p._cursor = 0;
  assert.strictEqual(p.triggerAt(10.5)?.time, 10);
});

ok('시간 오름차순 조회 1,000회가 선형 탐색과 같은 답을 낸다', () => {
  const p = make();
  const list = [];
  for (let i = 0; i < 200; i++) {
    const t = trig(+(i * 0.7 + Math.random() * 0.3).toFixed(4), 0.4 + Math.random());
    list.push(t);
    p.addTriggers([t]);
  }
  list.sort((a, b) => a.time - b.time);
  for (let k = 0; k < 1000; k++) {
    const target = +(k * 0.14).toFixed(4);   // 표시 시각처럼 단조 증가
    const fast = p.triggerAt(target);
    assert.strictEqual(fast?.time ?? null, brute(list, target)?.time ?? null,
      `target=${target} 에서 불일치`);
  }
});

console.log(`\n${pass}/7 통과`);
