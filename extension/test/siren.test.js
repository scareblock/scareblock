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

ok('격자 — 붙인 시각의 floor(t/0.48)·0.48부터 0.48초마다', async () => {
  const r = rig({ start: 10.13 });
  await runFor(r.sm, 4);
  assert.deepStrictEqual(r.asked.slice(0, 4), [10.08, 10.56, 11.04, 11.52]);
});

ok('트리거 — 넘는 패치만, [s, s+0.975), category siren', async () => {
  const r = rig({ start: 0, pfn: (s) => (s > 0.9 && s < 2 ? 0.2 : 0) });
  await runFor(r.sm, 6);
  assert.deepStrictEqual(r.trig.map((t) => +t.time.toFixed(2)), [0.96, 1.44, 1.92]);
  for (const t of r.trig) {
    assert.strictEqual(t.category, 'siren');
    assert.strictEqual(t.duration, 0.975);
    assert.strictEqual(t.source, 'yamnet-siren');
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
  assert.ok(r.asked[0] >= 29.76, `밀린 0초 패치를 묻지 않는다: ${r.asked[0]}`);
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
