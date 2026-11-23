/**
 * CLIP 전처리가 파이썬(PIL · eval/clip_zeroshot.preprocess)과 같은 값을 내는지 본다.
 *   node extension/test/clip-prep.test.js
 *
 * 무작위 RGB를 만들어 파이썬에 넘기고, PIL로 줄인 바이트와 최종 텐서를 받아 비교한다.
 * 파이썬(PIL · numpy)이 없으면 건너뛴다.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');
const { execFileSync } = require('child_process');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/clip-prep.js'), 'utf8'));
const { pilResize, clipPreprocess, clipResizeSize } = window.SB;

const PY = `
import sys, json, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
from eval.clip_zeroshot import preprocess, resize_size
w, h, f = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rgb = np.fromfile(f, np.uint8).reshape(h, w, 3)
meta = json.load(open(sys.argv[5]))
im = Image.fromarray(rgb)
r = np.asarray(im.resize(resize_size(w, h, meta["size"]), Image.BICUBIC))
r.tofile(f + ".resized")
preprocess(rgb, meta).astype(np.float32).tofile(f + ".tensor")
print(r.shape[1], r.shape[0])
`;

const ROOT = path.join(__dirname, '../..');
const META = path.join(ROOT, '_local/models/clip/meta.json');

let pass = 0, total = 0;
const ok = (name, fn) => {
  total++;
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
};

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

/** 매끄러운 무늬 + 잡음 — 실제 영상처럼 경계와 평탄한 곳이 섞이게 */
function image(w, h, seed) {
  const r = rng(seed), a = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    const v = 128 + 90 * Math.sin((x + 7 * c) / (9 + c)) * Math.cos(y / 13) + (r() - 0.5) * 80 + (x % 97 < 3 ? 120 : 0);
    a[(y * w + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(v)));
  }
  return a;
}

console.log('CLIP 전처리 = PIL');

let havePy = true;
try { execFileSync('python3', ['-c', 'import PIL, numpy'], { stdio: 'ignore' }); } catch { havePy = false; }
let skipped = '';
if (!havePy || !fs.existsSync(META)) {
  skipped = '파이썬(PIL · numpy) 또는 _local/models/clip/meta.json 없음';
  console.log(`  - ${skipped} — 건너뛴다`);
} else {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sb-clip-'));
  // 16:9가 아닌 것(854×480 · 640×480 · 1280×536 · 세로)은 반올림과 버림이 갈리는 크기다 (#51 리뷰 1)
  for (const [w, h, seed] of [[1280, 720, 1], [640, 360, 2], [854, 480, 3], [300, 500, 4], [224, 224, 5], [1920, 1080, 6],
    [640, 480, 7], [1280, 536, 8], [360, 640, 9]]) {
    const rgb = image(w, h, seed);
    const f = path.join(dir, `${w}x${h}.rgb`);
    fs.writeFileSync(f, rgb);
    const [pw, ph] = execFileSync('python3', ['-c', PY, ROOT, String(w), String(h), f, META]).toString().trim().split(' ').map(Number);
    ok(`${w}×${h} — 줄인 크기 · 바이트가 PIL과 같다`, () => {
      const [rw, rh] = clipResizeSize(w, h);
      assert.deepStrictEqual([rw, rh], [pw, ph]);
      const mine = pilResize(rgb, w, h, rw, rh);
      const ref = fs.readFileSync(f + '.resized');
      let diff = 0;
      for (let i = 0; i < mine.length; i++) if (mine[i] !== ref[i]) diff++;
      assert.strictEqual(diff, 0, `${diff}/${mine.length} 바이트가 다르다`);
    });
    ok(`${w}×${h} — 최종 텐서가 파이썬과 같다`, () => {
      const t = clipPreprocess(rgb, w, h);
      const ref = new Float32Array(fs.readFileSync(f + '.tensor').buffer.slice(0));
      assert.strictEqual(t.length, ref.length);
      let mx = 0;
      for (let i = 0; i < t.length; i++) mx = Math.max(mx, Math.abs(t[i] - ref[i]));
      assert.ok(mx < 1e-5, `최대 차이 ${mx}`);
    });
  }
  fs.rmSync(dir, { recursive: true, force: true });
}

// 건너뛰었으면 「통과」라고 쓰지 않는다 — 아무것도 안 본 것이다
console.log(skipped ? `\n${pass}/${total} 통과 · PIL 대조 건너뜀 (${skipped})` : `\n${pass}/${total} 통과`);
