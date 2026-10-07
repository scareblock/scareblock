/**
 * wav 하나를 SB.RuleCore에 2048 샘플씩 흘려 탐지를 JSON으로 낸다.
 * rule-parity.py가 부른다. 16-bit PCM wav만 읽는다 (eval/README.md의 추출 형식).
 *   node extension/test/rule-dump.js <wav>
 */
const fs = require('fs');
const path = require('path');

global.window = global;
eval(fs.readFileSync(path.join(__dirname, '../src/config.js'), 'utf8'));
eval(fs.readFileSync(path.join(__dirname, '../src/detector-rule.js'), 'utf8'));

const buf = fs.readFileSync(process.argv[2]);
let off = 12, sr = 0, ch = 1, bits = 0, data = null;
while (off + 8 <= buf.length) {
  const id = buf.toString('ascii', off, off + 4);
  const size = buf.readUInt32LE(off + 4);
  if (id === 'fmt ') {
    ch = buf.readUInt16LE(off + 10);
    sr = buf.readUInt32LE(off + 12);
    bits = buf.readUInt16LE(off + 22);
  } else if (id === 'data') {
    data = buf.subarray(off + 8, off + 8 + size);
  }
  off += 8 + size + (size & 1);
}
if (bits !== 16 || !data) throw new Error(`16-bit PCM만 읽는다 (bits=${bits})`);

const n = data.length / 2 / ch;
const x = new Float32Array(n);
for (let i = 0; i < n; i++) {
  let s = 0;
  for (let c = 0; c < ch; c++) s += data.readInt16LE((i * ch + c) * 2) / 32768;
  x[i] = s / ch;
}

const core = new window.SB.RuleCore(sr);
const out = [];
for (let k = 0; k < n; k += 2048) out.push(...core.push(x.subarray(k, Math.min(n, k + 2048))));
console.log(JSON.stringify(out));
