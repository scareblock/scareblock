/**
 * 사용자 설정 검증 — 값 정리 · 재생기/탐지기에 걸기 · 경고만 모드 조회 · 중계의 설정 전달.
 *   node extension/test/settings.test.js
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

const src = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
global.window = global;
for (const f of ['src/config.js', 'src/samples.js', 'src/settings.js', 'src/player.js']) eval(src(f));
const { Player } = window.SB;

let pass = 0, total = 0;
const tests = [];
const ok = (name, fn) => tests.push([name, fn]);

const trig = (time, category = 'jumpscare', duration = 1.5) =>
  ({ time, duration, category, confidence: 1, source: 'test' });

// ── 값 정리
ok('기본값 = 지금까지의 동작 (켜짐 · 블러 · 민감도 3 = surge 20 · 모두 켬)', () => {
  const s = SB.normalizeSettings(undefined);
  assert.deepStrictEqual(s, { enabled: true, mode: 'blur', sensitivity: 3,
    cats: { jumpscare: true, blood: true, spider: true, syringe: true, siren: true }, onboarded: false });
  assert.strictEqual(SB.surgeFor(s.sensitivity), 20);
  assert.strictEqual(SB.surgeFor(s.sensitivity), SB.RULE_PARAMS?.surge_of_quiet ?? 20);
});

ok('이상한 값은 기본값으로, 모르는 키는 버린다', () => {
  const s = SB.normalizeSettings({ enabled: 'yes', mode: 'hide', sensitivity: 9, cats: { blood: false, gore: false }, evil: 1 });
  assert.strictEqual(s.enabled, true);
  assert.strictEqual(s.mode, 'blur');
  assert.strictEqual(s.sensitivity, 3);
  assert.strictEqual(s.cats.blood, false);
  assert.ok(!('gore' in s.cats) && !('evil' in s));
  assert.strictEqual(SB.normalizeSettings({ sensitivity: 2.5 }).sensitivity, 3);
  assert.strictEqual(SB.normalizeSettings({ sensitivity: '5' }).sensitivity, 5);
});

ok('민감도 단계는 잰 발화율에 맞는다 — 다섯 단계 모두 잰 값', () => {
  assert.deepStrictEqual(SB.SENSITIVITY_SURGE, { 1: 60, 2: 35, 3: 20, 4: 10, 5: 5 });
  assert.deepStrictEqual(SB.SENSITIVITY_RATE, { 1: 0.49, 2: 1.18, 3: 2.19, 4: 3.94, 5: 6.37 });
  const r = [1, 2, 3, 4, 5].map((k) => SB.SENSITIVITY_RATE[k]);
  for (let i = 1; i < r.length; i++) assert.ok(r[i] > r[i - 1], '더 민감할수록 더 자주 울린다');
});

ok('CLIP은 피 · 거미 · 주사기 중 하나라도 켰을 때만 돈다', () => {
  const on = { ...SB.config, CLIP: true };
  assert.strictEqual(SB.clipWanted(SB.normalizeSettings(null), on), true);
  assert.strictEqual(SB.clipWanted(SB.normalizeSettings({ cats: { blood: false, spider: false, syringe: false } }), on), false);
  assert.strictEqual(SB.clipWanted(SB.normalizeSettings({ cats: { blood: false, spider: true, syringe: false } }), on), true);
  assert.strictEqual(SB.clipWanted(SB.normalizeSettings({ cats: { jumpscare: false } }), on), true, '점프 스케어는 CLIP과 무관');
  assert.strictEqual(SB.clipWanted(SB.normalizeSettings(null), { ...on, CLIP: false }), false, 'config가 끄면 끈다');
});

ok('민감도는 높을수록 surge가 작다(더 쉽게 울린다)', () => {
  const v = [1, 2, 3, 4, 5].map(SB.surgeFor);
  for (let i = 1; i < v.length; i++) assert.ok(v[i] < v[i - 1], JSON.stringify(v));
});

// ── 재생기 · 탐지기에 걸기
ok('applySettings — 모드 · 끈 종류 · 민감도를 건다. 같으면 아무것도 안 바꾼다', () => {
  const player = new Player({}, SB.config);
  let surge = 20;
  const detector = { core: { p: { surge_of_quiet: 20 } }, setSurge(n) { surge = n; this.core.p.surge_of_quiet = n; } };
  const s = SB.normalizeSettings({ mode: 'warn', sensitivity: 5, cats: { blood: false, spider: false } });
  const c = SB.applySettings(s, { player, detector });
  assert.strictEqual(player.mode, 'warn');
  assert.deepStrictEqual([...player.disabledCats].sort(), ['blood', 'spider']);
  assert.strictEqual(surge, 5);
  assert.strictEqual(c.length, 3, JSON.stringify(c));
  assert.deepStrictEqual(SB.applySettings(s, { player, detector }), []);
});

// ── 재생기 조회
ok('끈 종류는 triggerAt이 건너뛰고, 켠 종류는 그대로 잡는다', () => {
  const p = new Player({}, SB.config);
  p.addTriggers([trig(10, 'blood'), trig(10.5, 'jumpscare')]);
  p.disabledCats = new Set(['blood']);
  assert.strictEqual(p.triggerAt(10.2), null);
  assert.strictEqual(p.triggerAt(10.6).category, 'jumpscare');
  p.disabledCats = new Set();
  assert.strictEqual(p.triggerAt(10.6).category, 'blood');   // 커서를 지나치지 않았다
});

ok('수동(blurNow)은 끌 수 없다', () => {
  const p = new Player({}, SB.config);
  SB.applySettings(SB.normalizeSettings({ cats: { jumpscare: false, blood: false, spider: false, syringe: false } }), { player: p });
  p.addTriggers([trig(5, 'manual')]);
  assert.strictEqual(p.triggerAt(5.5).category, 'manual');
});

ok('upcoming — 곧 닿을 트리거를 찾되 커서를 옮기지 않는다', () => {
  const p = new Player({}, SB.config);
  p.addTriggers([trig(10), trig(12)]);
  assert.strictEqual(p.triggerAt(10.5).time, 10);
  assert.strictEqual(p.upcoming(10.5, 1.5).time, 12);
  assert.strictEqual(p.triggerAt(10.6).time, 10, 'upcoming 뒤에도 지금 트리거가 잡혀야 한다');
  assert.strictEqual(p.upcoming(8, 1.5), null);
  assert.strictEqual(p.upcoming(9, 1.5).time, 10);
});

ok('upcoming — 끈 종류는 미리 알리지 않는다', () => {
  const p = new Player({}, SB.config);
  p.addTriggers([trig(12, 'spider')]);
  p.disabledCats = new Set(['spider']);
  assert.strictEqual(p.upcoming(11, 1.5), null);
});

// ── 중계의 설정 전달 — 실제 ISOLATED world처럼 떨어진 vm에서
function bridgeWith(stored) {
  const msgs = [];
  const ls = [];
  const changeLs = [];
  const ctx = vm.createContext({
    location: { origin: 'https://www.youtube.com' }, btoa, console, setTimeout, Promise, Uint8Array, Float32Array,
    Number, Set, String, Error,
    chrome: {
      runtime: { sendMessage: async () => ({ ok: true }) },
      storage: {
        local: { get: async (k) => ({ [k]: stored.value }) },
        onChanged: { addListener: (f) => changeLs.push(f) },
      },
    },
    addEventListener: (t, f) => { if (t === 'message') ls.push(f); },
    postMessage: (d) => msgs.push(JSON.parse(JSON.stringify(d))),
  });
  const g = vm.runInContext('globalThis', ctx);
  ctx.window = g;
  vm.runInContext(src('src/bridge.js'), ctx);
  return {
    msgs: () => msgs.filter((m) => m.tag === SB.SETTINGS_TAG),
    ask: () => ls.forEach((f) => f({ source: g, data: { tag: SB.SETTINGS_TAG, dir: 'get' } })),
    change: (area) => changeLs.forEach((f) => f({ settings: {} }, area)),
  };
}
const flush = () => new Promise((r) => setImmediate(r));

ok('중계 — 처음에 저장된 설정을 한 번 보내고, 달라고 하면 또 보낸다', async () => {
  const stored = { value: { mode: 'warn' } };
  const b = bridgeWith(stored);
  await flush();
  assert.strictEqual(b.msgs().length, 1);
  assert.deepStrictEqual(b.msgs()[0], { tag: SB.SETTINGS_TAG, dir: 'push', settings: { mode: 'warn' } });
  b.ask();
  await flush();
  assert.strictEqual(b.msgs().length, 2);
});

ok('중계 — 저장소가 바뀌면 보내고, 다른 저장소(sync) 변화는 무시한다', async () => {
  const stored = { value: null };
  const b = bridgeWith(stored);
  await flush();
  assert.strictEqual(b.msgs()[0].settings, null);   // 저장된 게 없으면 null → 페이지는 기본값
  stored.value = { enabled: false };
  b.change('sync');
  await flush();
  assert.strictEqual(b.msgs().length, 1);
  b.change('local');
  await flush();
  assert.strictEqual(b.msgs().length, 2);
  assert.strictEqual(b.msgs()[1].settings.enabled, false);
});

ok('중계는 settings.js 없이 돈다(SB에 기대지 않는다)', () => {
  assert.ok(!/\bSB\./.test(src('src/bridge.js').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '')));
});

(async () => {
  console.log('사용자 설정');
  for (const [name, fn] of tests) {
    total++;
    try { await fn(); console.log(`  ✓ ${name}`); pass++; }
    catch (e) { console.error(`  ✗ ${name}\n    ${e.stack.split('\n').slice(0, 3).join('\n    ')}`); process.exitCode = 1; }
  }
  console.log(`${pass}/${total} 통과`);
})();
