/**
 * 사이렌 — 판정 · 0.48초 격자 · 트리거 · 다시 맞춤 · 설정.
 *   node extension/test/siren.test.js
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
const src = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
for (const f of ['src/config.js', 'src/samples.js', 'src/settings.js', 'src/siren.js']) eval(src(f));
SB.log = () => {};

const tests = [];
const ok = (name, fn) => tests.push([name, fn]);
const cfg = SB.config;

function scores(p, idx = 390) {
  const s = new Float32Array(521);
  s[idx] = p;
  return s;
}

ok('판정 — 4개 클래스 최댓값 ≥ θ면 사이렌, 391(공습경보)은 보지 않는다', () => {
  assert.deepStrictEqual(SB.SIREN_IDX, [390, 317, 318, 319]);
  assert.strictEqual(SB.sirenDecide(scores(cfg.SIREN_THETA)).hit, true);
  assert.strictEqual(SB.sirenDecide(scores(cfg.SIREN_THETA * 0.99)).hit, false);
  assert.strictEqual(SB.sirenDecide(scores(0.5, 318)).hit, true);         // 구급차
  assert.strictEqual(SB.sirenDecide(scores(0.9, 391)).hit, false);        // 공습경보는 등록에서 뺐다
  assert.ok(Math.abs(SB.sirenDecide(scores(0.3, 317)).p - 0.3) < 1e-6);
});

ok('θ는 #35 결과 그대로 (0.0025)', () => {
  assert.strictEqual(cfg.SIREN_THETA, 0.002500176429748535);
  assert.strictEqual(cfg.SIREN_HOP_S, 0.48);
});

/** 가짜 영상 · YAMNet. step()마다 시각이 dt만큼 간다. 패치 점수는 시각 함수 */
function rig({ start = 10.13, pfn = () => 0, nullAt = new Set() } = {}) {
  const video = { currentTime: start, paused: false };
  const asked = [];
  const yam = {
    async scoresAt(s) {
      await new Promise(setImmediate);   // 실제처럼 한 박자 쉰다 — 안 그러면 루프가 마이크로태스크만 돌아 테스트가 끼어들 틈이 없다
      asked.push(+s.toFixed(2));
      // 패치가 찰 때까지 시간이 흐른 것처럼 — 창 끝까지 보낸다
      if (video.currentTime < s + 0.975) video.currentTime = s + 0.975 + 0.01;
      if (nullAt.has(+s.toFixed(2))) return null;
      return scores(pfn(s));
    },
  };
  const trig = [];
  const player = { addTriggers: (l) => trig.push(...l) };
  const sm = new SB.SirenSampler(video, player, () => yam, cfg);
  return { video, asked, trig, sm };
}

async function runFor(sm, n) {
  const orig = SB.sleep;
  SB.sleep = () => Promise.resolve();
  sm.start();
  for (let i = 0; i < n * 5 && sm.n.패치 + sm.n.못얻음 < n; i++) await new Promise(setImmediate);
  sm.stop();
  SB.sleep = orig;
}

ok('격자 — 지금 판정할 수 있는 최근 격자 floor((t−0.975)/0.48)·0.48부터 0.48초마다', async () => {
  const r = rig({ start: 10.13 });
  await runFor(r.sm, 4);
  assert.deepStrictEqual(r.asked.slice(0, 4), [9.12, 9.6, 10.08, 10.56]);
});

ok('트리거 — 넘는 패치만, [s, s+0.975), category siren', async () => {
  const r = rig({ start: 0, pfn: (s) => (s > 0.9 && s < 2 ? 0.2 : 0) });
  await runFor(r.sm, 6);
  assert.deepStrictEqual(r.trig.map((t) => +t.time.toFixed(2)), [0.96, 1.44, 1.92]);
  for (const t of r.trig) {
    assert.strictEqual(t.category, 'siren');
    assert.strictEqual(t.duration, 0.975);
    assert.strictEqual(t.source, 'yamnet-siren');
    assert.strictEqual(t.confidence, 1);   // 점수(0.00x)를 그대로 넣지 않는다 (#70 리뷰)
  }
  assert.strictEqual(r.sm.n.사이렌, 3);
});

ok('패치를 못 얻으면(시크 등) 가리지 않고 지금 위치로 다시 맞춘다', async () => {
  const r = rig({ start: 0, pfn: () => 0.9, nullAt: new Set([0.48]) });
  await runFor(r.sm, 4);
  assert.strictEqual(r.sm.n.못얻음, 1);
  assert.ok(!r.trig.some((t) => Math.abs(t.time - 0.48) < 1e-9), '못 얻은 패치로는 트리거를 내지 않는다');
});

ok('이미 화면을 지나간 패치는 건너뛰고 다시 맞춘다', async () => {
  const r = rig({ start: 0 });
  r.sm.running = true;
  r.sm._anchor();
  r.video.currentTime = 30;              // 탭 숨김 등으로 30초 밀림
  const orig = SB.sleep; SB.sleep = () => Promise.resolve();
  r.sm._loop();
  for (let i = 0; i < 10 && r.sm.n.패치 < 1; i++) await new Promise(setImmediate);
  r.sm.stop(); SB.sleep = orig;
  assert.ok(r.sm.n.다시맞춤 >= 1);
  assert.ok(r.asked[0] >= 28.8, `밀린 0초 패치를 묻지 않는다: ${r.asked[0]}`);
});

ok('결과가 화면에 닿기 전에 못 올 패치는 건너뛴다 (#70 리뷰 🟡1)', async () => {
  const r = rig({ start: 0 });
  r.sm.running = true;
  r.sm.next = 0.48;
  r.video.currentTime = 0.48 + cfg.DELAY_SEC - SB.SIREN_LATE_MARGIN_S + 0.05;   // 여유보다 늦었다
  const orig = SB.sleep; SB.sleep = () => Promise.resolve();
  r.sm._loop();
  for (let i = 0; i < 10 && r.sm.n.패치 < 1; i++) await new Promise(setImmediate);
  r.sm.stop(); SB.sleep = orig;
  assert.ok(r.asked[0] > 0.48, `늦을 패치(0.48)를 묻지 않는다: ${r.asked[0]}`);
  assert.strictEqual(r.sm.n.다시맞춤, 1);
});

ok('다시 맞출 때 탭이 들고 있는 구간보다 앞 격자는 고르지 않는다 (#70 리뷰 🟢3)', () => {
  const video = { currentTime: 50.3, paused: false };
  const yam = { tap: { core: { span: () => ({ from: 49.9, to: 50.3 }) } }, scoresAt: async () => null };
  const sm = new SB.SirenSampler(video, { addTriggers() {} }, () => yam, cfg);
  sm._anchor();
  // floor((50.3−0.975)/0.48)·0.48 = 48.96 < 49.9 → 49.9 뒤 첫 격자 49.92
  assert.ok(Math.abs(sm.next - 49.92) < 1e-9, `${sm.next}`);
});

ok('여유 — 최근 계산 시간 p90, 0.3~1.0초로 자른다 · 결과 도착 기준으로 늦음을 본다 (#70 재리뷰 🟢2)', () => {
  const sm = new SB.SirenSampler({ currentTime: 0, paused: false }, { addTriggers() {} }, () => null, cfg);
  assert.strictEqual(sm.margin, SB.SIREN_LATE_MARGIN_S);
  for (let i = 0; i < 20; i++) sm.lag.push(0.05);
  sm._updateMargin();
  assert.strictEqual(sm.margin, 0.3, '빠르면 바닥 0.3');
  for (let i = 0; i < 64; i++) sm.lag.push(0.7);
  sm._updateMargin();
  assert.ok(Math.abs(sm.margin - 0.7) < 1e-9, `${sm.margin}`);
  for (let i = 0; i < 64; i++) sm.lag.push(5);
  sm._updateMargin();
  assert.strictEqual(sm.margin, SB.SIREN_LATE_MARGIN_MAX_S, '천장 1.0');
  // 천장에서도 다시 맞춘 직후 패치(밀림 ≤ 1.455초)는 늦지 않다 — 또 건너뛰기 반복이 없다
  assert.strictEqual(sm._late(10, 10 + 0.975 + 0.48), false);
  sm.margin = 0.7;
  assert.strictEqual(sm._late(10, 12.2), false);
  assert.strictEqual(sm._late(10, 12.4), true, '시작은 제때여도 결과가 늦을 패치');
});

ok('뒤로 시크해서 패치를 못 얻으면 시크 전 위치가 아니라 새 격자로 (#70 재리뷰 🟢3)', async () => {
  const video = { currentTime: 20, paused: false };
  const asked = [];
  let first = true;
  const yam = {
    async scoresAt(s) {
      await new Promise(setImmediate);
      asked.push(+s.toFixed(2));
      if (first) { first = false; video.currentTime = 5; return null; }   // 묻는 사이 5초로 되감음
      return scores(0);
    },
  };
  const sm = new SB.SirenSampler(video, { addTriggers() {} }, () => yam, cfg);
  await runFor(sm, 2);
  assert.ok(asked[1] < 5, `되감은 뒤 격자를 물어야 한다: ${asked}`);
  assert.strictEqual(sm.n.다시맞춤, 0, '다시맞춤이 더 오르지 않는다');
});

ok('보고 — 창 끝 뒤 지연 · 화면까지 남은 시간', async () => {
  const r = rig({ start: 0 });
  await runFor(r.sm, 3);
  const rep = r.sm.report();
  assert.strictEqual(rep.패치, 3);
  assert.ok(rep.창끝_뒤_s && rep.화면까지_남은_s);
});

ok('설정 — 사이렌 칸 기본 켬, 끄면 sirenWanted false', () => {
  const s = SB.normalizeSettings(null);
  assert.strictEqual(s.cats.siren, true);
  assert.strictEqual(SB.CATEGORY_LABEL.siren, '사이렌');
  assert.strictEqual(SB.sirenWanted(s), true);
  assert.strictEqual(SB.sirenWanted(SB.normalizeSettings({ cats: { siren: false } })), false);
  assert.strictEqual(SB.sirenWanted(s, { ...cfg, SIREN: false }), false);
});

(async () => {
  let pass = 0;
  console.log('사이렌');
  for (const [name, fn] of tests) {
    try { await fn(); console.log(`  ✓ ${name}`); pass++; } catch (e) { console.log(`  ✗ ${name}\n    ${e.message}`); }
  }
  console.log(`${pass}/${tests.length} 통과`);
  process.exit(pass === tests.length ? 0 : 1);
})();
