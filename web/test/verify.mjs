// Compare the wasm port against the PyTorch reference, then measure flight performance.
import fs from 'node:fs';
const SEED = +(process.env.SEED || 7);
const wasmPath = process.argv[2], polPath = process.argv[3], refPath = process.argv[4];
const { instance } = await WebAssembly.instantiate(fs.readFileSync(wasmPath));
const W = instance.exports;
const f32 = () => new Float32Array(W.memory.buffer);
function f16to32(u16) { const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) { const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); } return out; }
function loadPolicy(p) {
  const bytes = Buffer.from(p.data, 'base64');
  const arr = p.dtype === 'f16' ? f16to32(new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2)) : new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
  const ptr = W.alloc_f32(arr.length); f32().set(arr, ptr / 4);
  return W.load_policy(ptr, arr.length);
}
const pols = JSON.parse(fs.readFileSync(polPath));
if (refPath) {
  const ref = JSON.parse(fs.readFileSync(refPath));
  console.log('load_policy ->', loadPolicy(pols[0]));
  W.set_ring_radius(2.0);
  W.init(4, 1);
  const buf = W.alloc_f32(28);
  ref.init.forEach((s, i) => { f32().set(s, buf / 4); W.debug_set(i, buf); });
  let maxObs = 0;
  for (let i = 0; i < 4; i++) { const o = f32().subarray(W.debug_obs(i) / 4, W.debug_obs(i) / 4 + 38);
    ref.obs0[i].forEach((v, k) => maxObs = Math.max(maxObs, Math.abs(v - o[k]))); }
  console.log('max |obs diff| at t=0:', maxObs.toExponential(2));
  const S = W.stride();
  const firstPass = [0, 1, 2, 3].map(i => { const k = ref.passed.findIndex(p => p[i]); return k < 0 ? 999 : k; });
  const report = [10, 25, 50, 100, 149];
  for (let t = 0; t < 150; t++) {
    W.step(1);
    if (report.includes(t)) {
      const st = f32().subarray(W.state_ptr() / 4);
      const errs = [0, 1, 2, 3].map(i => t < firstPass[i] ? Math.hypot(...[0, 1, 2].map(a => st[i * S + a] - ref.pos[t][i][a])).toFixed(4) : 'ring');
      console.log(`step ${t + 1} (${((t + 1) * 0.02).toFixed(2)}s) pos error per bird [m]:`, errs.join('  '));
    }
  }
}
// performance: N birds, 60 s, rings per minute
for (const p of pols) {
  loadPolicy(p); W.set_ring_radius(2.0); W.init(32, SEED);
  const S = W.stride(); let crashes = 0, prev = new Float32Array(32);
  const t0 = performance.now();
  for (let t = 0; t < 3000; t++) { W.step(1); const st = f32().subarray(W.state_ptr() / 4);
    for (let i = 0; i < 32; i++) { const c = st[i * S + 26]; if (c > 0 && prev[i] === 0) crashes++; prev[i] = c; } }
  const ms = performance.now() - t0;
  const st = f32().subarray(W.state_ptr() / 4);
  let total = 0; for (let i = 0; i < 32; i++) total += st[i * S + 27];
  console.log(`${p.name} [${p.dtype}]: ${(total / 32).toFixed(1)} rings/bird/min, ${crashes} crashes / 32 birds / 60s | sim speed ${(32 * 3000 / ms * 1000 / 50).toFixed(0)}x realtime-bird`);
}
