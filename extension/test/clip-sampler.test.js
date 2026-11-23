/**
 * CLIP 시각 경로 — 점수 계산(heads)과 트리거 판정을 본다.
 *   node extension/test/clip-sampler.test.js
 *
 * 1) 판정: θ 이상이면 [t, t+1) 트리거, 피는 보강 모델이 있으면 그것과 그 θ를 쓴다.
 * 2) heads 일치: 오프라인에서 저장한 임베딩(_local/…/clip-emb)을 넣으면 eval이 저장한 점수(clip-scores ·
 *    probe-scores)와 같은 값이 나오는가. 데이터나 models/clip_heads.json이 없으면 건너뛴다.
 *    전처리 일치는 clip-prep.test.js가 본다 — 둘이 합쳐 「같은 프레임 → 같은 점수」가 된다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

global.window = global;
const src = (f) => fs.readFileSync(path.join(__dirname, '..', f), 'utf8');
for (const f of ['src/config.js', 'src/clip-prep.js', 'src/clip-sampler.js']) eval(src(f));

let pass = 0, total = 0;
const ok = (name, fn) => {
  total++;
  try { fn(); console.log(`  ✓ ${name}`); pass++; } catch (e) { console.log(`  ✗ ${name}\n    ${e.message}`); }
};
let skipped = 0;
const skip = (name, why) => { skipped++; console.log(`  - ${name} (건너뜀: ${why})`); };

console.log('CLIP 시각 경로');

const cfg = SB.config;

ok('판정 — θ 이상인 종류만, [t, t+1) 트리거로', () => {
  const r = { zs: { blood: 0.9, spider: cfg.CLIP_THETA.spider, syringe: 0.001 }, probe: {} };
  const tr = SB.clipDecide(r, 12, cfg);
  assert.deepStrictEqual(tr.map((x) => x.category).sort(), ['blood', 'spider']);
  for (const x of tr) {
    assert.strictEqual(x.time, 12);
    assert.strictEqual(x.duration, 1);
    assert.strictEqual(x.source, 'clip-zeroshot');
  }
});

ok('판정 — 피는 보강 모델이 있으면 zero-shot을 무시하고 보강 모델의 θ로', () => {
  const lo = SB.clipDecide({ zs: { blood: 0.99 }, probe: { blood: cfg.CLIP_PROBE_THETA.blood * 0.9 } }, 0, cfg);
  assert.strictEqual(lo.length, 0);
  const hi = SB.clipDecide({ zs: { blood: 0.0 }, probe: { blood: cfg.CLIP_PROBE_THETA.blood } }, 0, cfg);
  assert.strictEqual(hi.length, 1);
  assert.strictEqual(hi[0].source, 'clip-probe');
});

ok('판정 — 응답이 없으면 아무것도 가리지 않는다', () => {
  assert.deepStrictEqual(SB.clipDecide(null, 0, cfg), []);
  assert.deepStrictEqual(SB.clipDecide({ zs: {}, probe: {} }, 0, cfg), []);
});

ok('판정 — 꺼 둔 종류는 보지 않는다', () => {
  const c2 = { ...cfg, CLIP_CATS: ['spider'] };
  assert.strictEqual(SB.clipDecide({ zs: { blood: 1, spider: 0 }, probe: {} }, 0, c2).length, 0);
});

ok('rgbaToRgb · clipCrop — 크기와 채널', () => {
  const rgba = new Uint8ClampedArray([1, 2, 3, 255, 4, 5, 6, 255]);
  assert.deepStrictEqual(Array.from(SB.rgbaToRgb(rgba)), [1, 2, 3, 4, 5, 6]);
  const crop = SB.clipCrop(new Uint8Array(320 * 240 * 3).fill(7), 320, 240);
  assert.strictEqual(crop.length, 224 * 224 * 3);
  assert.ok(crop.every((v) => v === 7));
  const t = SB.clipNormalize(crop);
  assert.ok(Math.abs(t[0] - (7 / 255 - SB.CLIP_MEAN[0]) / SB.CLIP_STD[0]) < 1e-6);
});

// ── 오프라인 점수와 같은가 ───────────────────────────────────────────
const ROOT = path.join(__dirname, '../..');
const HEADS = path.join(__dirname, '../models/clip_heads.json');
// 리사이즈를 버림으로 고친 뒤(#51 리뷰 1) 다시 낸 임베딩 · 점수. fetch-deps.sh의 보강 모델도 여기 probe.json이다
const DATA = path.join(ROOT, '_local/dataset/resize-1123');
const EMB = path.join(DATA, 'clip-emb');
const ZS = path.join(DATA, 'clip-scores');
const PR = path.join(DATA, 'probe-scores');

/** 아주 작은 .npy 읽기 — float32 2차원 (C 순서) */
function readNpy(f) {
  const b = fs.readFileSync(f);
  const hl = b.readUInt16LE(8);
  const hdr = b.slice(10, 10 + hl).toString('latin1');
  assert.ok(/'descr': '<f4'/.test(hdr) && /'fortran_order': False/.test(hdr), hdr);
  const shape = hdr.match(/'shape': \(([^)]*)\)/)[1].split(',').filter((s) => s.trim()).map(Number);
  const off = 10 + hl;
  const n = shape.reduce((x, y) => x * y, 1);
  const data = new Float32Array(b.buffer.slice(b.byteOffset + off, b.byteOffset + off + n * 4));
  return { shape, data };
}

/** .npz 안의 p.npy만 꺼낸다. numpy savez는 압축하지 않으므로(zip64 머리말이라 크기 칸이 비어 있어)
 *  파일 이름 뒤 첫 NUMPY 머리말을 찾아 shape로 길이를 정한다 */
function readNpzP(f) {
  const b = fs.readFileSync(f);
  const at = b.indexOf(Buffer.from('p.npy'));
  assert.ok(at >= 0, `${f}에 p.npy 없음`);
  const m = b.indexOf(Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]), at);   // \x93NUMPY
  const hl = b.readUInt16LE(m + 8);
  const hdr = b.slice(m + 10, m + 10 + hl).toString('latin1');
  const shape = hdr.match(/'shape': \(([^)]*)\)/)[1].split(',').filter((x) => x.trim()).map(Number);
  const dt = hdr.match(/'descr': '([^']*)'/)[1];
  const size = dt === '<f8' ? 8 : 4;
  const st = b.byteOffset + m + 10 + hl;
  const ab = b.buffer.slice(st, st + shape.reduce((x, y) => x * y, 1) * size);
  return { shape, data: size === 8 ? new Float64Array(ab) : new Float32Array(ab) };
}

if (!fs.existsSync(HEADS)) skip('heads 일치', 'models/clip_heads.json 없음 — fetch-deps.sh');
else if (!fs.existsSync(EMB)) skip('heads 일치', '_local 임베딩 없음');
else {
  const heads = JSON.parse(fs.readFileSync(HEADS, 'utf8'));
  const CATS = ['blood', 'syringe', 'spider'];
  const all = fs.readdirSync(EMB).filter((f) => f.endsWith('.npy')).map((f) => f.slice(0, -4));
  // 앞 8개 + 16:9가 아닌 클립(리사이즈가 바뀐 것) — c034는 피 보강 모델의 양성 클립이다
  const ids = [...new Set([...all.slice(0, 8), ...['c034', 'c041'].filter((c) => all.includes(c))])];

  ok(`zero-shot — eval 점수와 같다 (클립 ${ids.length}개)`, () => {
    let worst = 0, n = 0;
    for (const id of ids) {
      const e = readNpy(path.join(EMB, `${id}.npy`));
      const p = readNpzP(path.join(ZS, `${id}.npz`));
      const [rows, d] = e.shape;
      assert.strictEqual(p.shape[0], rows, `${id} 프레임 수`);
      for (let r = 0; r < rows; r++) {
        const { zs } = SB.clipHeads(e.data.subarray(r * d, (r + 1) * d), heads);
        CATS.forEach((c, k) => { worst = Math.max(worst, Math.abs(zs[c] - p.data[r * 3 + k])); });
        n++;
      }
    }
    assert.ok(worst < 1e-3, `최대 차이 ${worst}`);
    console.log(`    프레임 ${n}장 · 최대 차이 ${worst.toExponential(2)}`);
  });

  if (!heads.probe || !fs.existsSync(PR)) skip('보강 모델 일치', '보강 모델 · probe-scores 없음');
  else {
    ok('보강 모델 — eval 점수와 같다', () => {
      let worst = 0, n = 0;
      for (const id of ids) {
        const f = path.join(PR, `${id}.npz`);
        if (!fs.existsSync(f)) continue;
        const e = readNpy(path.join(EMB, `${id}.npy`));
        const p = readNpzP(f);
        const d = e.shape[1];
        for (let r = 0; r < e.shape[0]; r++) {
          const { probe } = SB.clipHeads(e.data.subarray(r * d, (r + 1) * d), heads);
          worst = Math.max(worst, Math.abs(probe.blood - p.data[r * 3]));
          n++;
        }
      }
      assert.ok(n > 0, '비교한 프레임이 없다');
      assert.ok(worst < 1e-4, `최대 차이 ${worst}`);
      console.log(`    프레임 ${n}장 · 최대 차이 ${worst.toExponential(2)}`);
    });
  }
}

console.log(skipped ? `${pass}/${total} 통과 · ${skipped}개 건너뜀` : `${pass}/${total} 통과`);
process.exit(pass === total ? 0 : 1);
