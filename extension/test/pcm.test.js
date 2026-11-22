/**
 * YAMNet 입력 만들기 검증.
 *   node extension/test/pcm.test.js
 *
 * 리샘플러가 시각을 밀지 않는지(패치를 탐지 시각으로 자르므로), 8 kHz 넘는 소리를 접어 넣지 않는지,
 * 콜백 크기에 따라 결과가 달라지지 않는지, 고리가 미디어 시각으로 맞는 구간을 꺼내는지 본다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/pcm.js'), 'utf8'));
const { Resampler, TapCore, pcm } = window.SB;

let pass = 0, total = 0;
const ok = (name, fn) => {
  total++;
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
};

const tone = (sr, f, sec, amp = 0.5) =>
  Float32Array.from({ length: Math.round(sr * sec) }, (_, i) => amp * Math.sin(2 * Math.PI * f * i / sr));
const rms = (x, a = 0, b = x.length) => {
  let s = 0;
  for (let i = a; i < b; i++) s += x[i] * x[i];
  return Math.sqrt(s / (b - a));
};
const pushAll = (rs, x, chunk) => {
  const out = [];
  for (let i = 0; i < x.length; i += chunk) out.push(...rs.push(x.subarray(i, i + chunk)));
  return Float32Array.from(out);
};

console.log('YAMNet 입력');

for (const sr of [48000, 44100]) {
  ok(`${sr} Hz — 1 kHz 사인은 크기 그대로 지나간다`, () => {
    const y = pushAll(new Resampler(sr), tone(sr, 1000, 1), 2048);
    const r = rms(y, 2000, y.length - 2000);
    assert.ok(Math.abs(r - 0.5 / Math.SQRT2) < 0.005, `rms=${r}`);
  });

  ok(`${sr} Hz — 11 kHz 사인은 접혀 들어오지 않는다 (-40 dB 아래)`, () => {
    const y = pushAll(new Resampler(sr), tone(sr, 11000, 1), 2048);
    const r = rms(y, 2000, y.length - 2000) / (0.5 / Math.SQRT2);
    assert.ok(r < 0.01, `상대 rms=${r}`);
  });

  ok(`${sr} Hz — 출력 k는 입력 시각 k/16000초다 (충격의 위치가 밀리지 않는다)`, () => {
    const x = new Float32Array(sr);
    const t = 0.4321;
    x[Math.round(t * sr)] = 1;
    const y = pushAll(new Resampler(sr), x, 2048);
    let best = 0;
    for (let i = 1; i < y.length; i++) if (Math.abs(y[i]) > Math.abs(y[best])) best = i;
    assert.ok(Math.abs(best / 16000 - t) <= 1 / 16000 + 1e-9, `봉우리 ${best / 16000}s, 기대 ${t}s`);
  });
}

ok('콜백 크기가 달라도 같은 출력이다', () => {
  const x = tone(44100, 523, 0.7);
  const a = pushAll(new Resampler(44100), x, 2048);
  const b = pushAll(new Resampler(44100), x, 37);
  assert.strictEqual(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-6, `i=${i}`);
});

ok('입력 1초 → 출력은 반폭만큼 덜 나온 뒤, 다음 입력에서 이어진다', () => {
  const rs = new Resampler(48000);
  const n1 = rs.push(new Float32Array(48000)).length;
  assert.ok(n1 < 16000 && n1 > 15900, `n1=${n1}`);
  const n2 = rs.push(new Float32Array(48000)).length;
  assert.strictEqual(n1 + n2, 32000 - (16000 - n1));
});

ok('base64는 손실이 없다', () => {
  const x = Float32Array.from([0, -0, 1, -1, 1e-30, 3.14159, NaN, Infinity, -0.5]);
  const y = pcm.decode(pcm.encode(x));
  assert.strictEqual(y.length, x.length);
  for (let i = 0; i < x.length; i++) assert.ok(Object.is(x[i], y[i]), `i=${i}: ${x[i]} vs ${y[i]}`);
  const big = tone(16000, 440, 0.975);
  const z = pcm.decode(pcm.encode(big));
  assert.deepStrictEqual(Array.from(z), Array.from(big));
});

ok('고리 — 미디어 시각으로 패치를 꺼낸다', () => {
  const tap = new TapCore(48000, 4);
  tap.setAnchor(100, 1);
  // 100.5초에 충격 하나
  const x = new Float32Array(48000 * 3);
  x[Math.round(0.5 * 48000)] = 1;
  for (let i = 0; i < x.length; i += 2048) tap.push(x.subarray(i, i + 2048));
  const s = tap.slice(100.25);
  assert.ok(s.pcm, JSON.stringify(Object.keys(s)));
  assert.strictEqual(s.pcm.length, 15600);
  let best = 0;
  for (let i = 1; i < s.pcm.length; i++) if (Math.abs(s.pcm[i]) > Math.abs(s.pcm[best])) best = i;
  assert.ok(Math.abs(best - 0.25 * 16000) <= 1, `봉우리 ${best}`);
});

ok('고리 — 아직 안 들어온 구간은 wait, 밀려난 구간은 gone', () => {
  const tap = new TapCore(48000, 2);
  assert.ok(tap.slice(0).wait);                       // 원점 없음
  tap.setAnchor(10, 1);
  tap.push(new Float32Array(48000 * 3));               // 10 ~ 13초
  assert.ok(tap.slice(12.5).wait);                    // 12.5 + 0.975 > 13
  assert.ok(tap.slice(10.2).gone);                    // 2초만 들고 있다
  assert.ok(tap.slice(11.5).pcm);
  const sp = tap.span();
  assert.ok(Math.abs(sp.to - 13) < 0.01 && Math.abs(sp.from - 11) < 0.01, JSON.stringify(sp));
});

ok('고리 — 2배속이면 오디오 1초가 미디어 2초다', () => {
  const tap = new TapCore(48000, 8);
  tap.setAnchor(0, 2);
  const x = new Float32Array(48000 * 2);              // 오디오 2초 = 미디어 4초
  x[Math.round(1.0 * 48000)] = 1;                     // 오디오 1초 = 미디어 2초
  tap.push(x);
  const s = tap.slice(1.5);                           // 미디어 1.5 = 오디오 0.75
  let best = 0;
  for (let i = 1; i < s.pcm.length; i++) if (Math.abs(s.pcm[i]) > Math.abs(s.pcm[best])) best = i;
  assert.ok(Math.abs(best - 0.25 * 16000) <= 1, `봉우리 ${best}`);
});

ok('고리 — reset() 뒤에는 원점을 다시 잡을 때까지 wait', () => {
  const tap = new TapCore(48000, 4);
  tap.setAnchor(0, 1);
  tap.push(new Float32Array(48000 * 2));
  const g = tap.gen;
  tap.reset();
  assert.ok(tap.slice(0.1).wait);
  assert.notStrictEqual(tap.gen, g);
});

console.log(`${pass}/${total} 통과`);
