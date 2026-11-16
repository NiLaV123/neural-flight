//! Bird flight simulator + policy inference, a faithful port of bird_env.py / train.py
//! compiled to WebAssembly. No dependencies; plain C ABI exports driven from JavaScript.
#![allow(static_mut_refs)]

use std::f32::consts::PI;

const RHO: f32 = 1.225;
const G: f32 = 9.81;
const MASS: f32 = 0.8;
const INERTIA: [f32; 3] = [0.030, 0.035, 0.060];
const N_SEG: usize = 3;
const N_SURF: usize = 2 * N_SEG + 1;
const HALF_SPAN: f32 = 0.6;
const CHORD: f32 = 0.20;
const SHOULDER: [f32; 3] = [0.0, 0.06, 0.02];
const TAIL_POS: [f32; 3] = [-0.38, 0.0, 0.0];
const TAIL_AREA: (f32, f32) = (0.02, 0.06);
const BODY_CDA: f32 = 0.006;
const J_LO: [f32; 9] = [-0.7, -0.45, -0.6, -0.7, -0.45, -0.6, -0.7, -0.8, 0.0];
const J_HI: [f32; 9] = [1.0, 0.45, 0.6, 1.0, 0.45, 0.6, 0.7, 0.8, 1.0];
const J_RATE: [f32; 9] = [18.0, 8.0, 12.0, 18.0, 8.0, 12.0, 8.0, 8.0, 6.0];
const WING_INCIDENCE: f32 = 0.06;
pub const DT: f32 = 0.02;
const SUBSTEPS: usize = 4;
const RING_TIMEOUT: u32 = 600;
pub const N_OBS: usize = 38;
pub const N_ACT: usize = 9;
/// floats per bird in the shared render buffer
pub const STRIDE: usize = 32;

type V3 = [f32; 3];
type M3 = [[f32; 3]; 3];

#[inline] fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
#[inline] fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
#[inline] fn scale(a: V3, s: f32) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
#[inline] fn dot(a: V3, b: V3) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
#[inline] fn norm(a: V3) -> f32 { dot(a, a).sqrt() }
#[inline] fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline] fn mv(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
#[inline] fn mtv(m: &M3, v: V3) -> V3 { // m^T v
    [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
     m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
     m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]]
}
fn mm(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 { for j in 0..3 { r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]; } }
    r
}
fn col(m: &M3, j: usize) -> V3 { [m[0][j], m[1][j], m[2][j]] }
fn rot_x(a: f32) -> M3 { let (s, c) = a.sin_cos(); [[1., 0., 0.], [0., c, -s], [0., s, c]] }
fn rot_y(a: f32) -> M3 { let (s, c) = a.sin_cos(); [[c, 0., s], [0., 1., 0.], [-s, 0., c]] }
fn rot_z(a: f32) -> M3 { let (s, c) = a.sin_cos(); [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]] }

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [aw * bw - ax * bx - ay * by - az * bz,
     aw * bx + ax * bw + ay * bz - az * by,
     aw * by - ax * bz + ay * bw + az * bx,
     aw * bz + ax * by - ay * bx + az * bw]
}
fn quat_to_mat(q: [f32; 4]) -> M3 {
    let [w, x, y, z] = q;
    [[1. - 2. * (y * y + z * z), 2. * (x * y - w * z), 2. * (x * z + w * y)],
     [2. * (x * y + w * z), 1. - 2. * (x * x + z * z), 2. * (y * z - w * x)],
     [2. * (x * z - w * y), 2. * (y * z + w * x), 1. - 2. * (x * x + y * y)]]
}

// ------------------------------------------------------------------ rng
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 { // uniform [0,1)
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn sym(&mut self) -> f32 { self.next() * 2.0 - 1.0 }
    fn gauss(&mut self) -> f32 {
        let u1 = self.next().max(1e-7); let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ------------------------------------------------------------------ policy MLP
struct Layer { w: Vec<f32>, b: Vec<f32>, n_in: usize, n_out: usize }
struct Policy { mean: Vec<f32>, std: Vec<f32>, layers: Vec<Layer>, log_std: Vec<f32> }

impl Policy {
    fn act(&self, obs: &[f32; N_OBS], buf_a: &mut Vec<f32>, buf_b: &mut Vec<f32>) -> [f32; N_ACT] {
        buf_a.clear();
        for i in 0..N_OBS { buf_a.push(((obs[i] - self.mean[i]) / (self.std[i] + 1e-6)).clamp(-8.0, 8.0)); }
        let nl = self.layers.len();
        for (li, l) in self.layers.iter().enumerate() {
            buf_b.clear();
            for o in 0..l.n_out {
                let row = &l.w[o * l.n_in..(o + 1) * l.n_in];
                let mut acc = l.b[o] + dotf(row, &buf_a[..l.n_in]);
                if li + 1 < nl && acc < 0.0 { acc = acc.exp() - 1.0; } // ELU
                buf_b.push(acc);
            }
            std::mem::swap(buf_a, buf_b);
        }
        let mut a = [0.0; N_ACT];
        a.copy_from_slice(&buf_a[..N_ACT]);
        a
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn dotf(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;
    let n = a.len().min(b.len());
    let (mut s0, mut s1) = (f32x4_splat(0.0), f32x4_splat(0.0));
    let mut i = 0;
    unsafe {
        while i + 8 <= n {
            let pa = a.as_ptr().add(i) as *const v128;
            let pb = b.as_ptr().add(i) as *const v128;
            s0 = f32x4_add(s0, f32x4_mul(v128_load(pa), v128_load(pb)));
            s1 = f32x4_add(s1, f32x4_mul(v128_load(pa.add(1)), v128_load(pb.add(1))));
            i += 8;
        }
    }
    let s = f32x4_add(s0, s1);
    let mut acc = f32x4_extract_lane::<0>(s) + f32x4_extract_lane::<1>(s) + f32x4_extract_lane::<2>(s) + f32x4_extract_lane::<3>(s);
    while i < n { acc += a[i] * b[i]; i += 1; }
    acc
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn dotf(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

// ------------------------------------------------------------------ bird
#[derive(Clone, Copy, Default)]
struct Surf { r: [V3; N_SURF], c: [V3; N_SURF], n: [V3; N_SURF], s: [V3; N_SURF], area: [f32; N_SURF] }

#[derive(Clone, Copy, Default)]
struct Bird {
    pos: V3, vel: V3, quat: [f32; 4], omega: V3,
    joints: [f32; 9], target: [f32; 9],
    ring_pos: V3, ring_n: V3,
    rings: u32, since_ring: u32,
    crashed: f32, // >0: seconds since crash (frozen, waiting to respawn)
    total_rings: u32,
    passed_flag: f32,
    last_ring: V3,
    speed_air: f32,
}

fn seg_geom() -> ([f32; N_SEG], [f32; N_SEG]) {
    let mut r = [0.0; N_SEG];
    let mut area = [0.0; N_SEG];
    let mut sum = 0.0;
    for k in 0..N_SEG {
        let frac = (k as f32 + 0.5) / N_SEG as f32;
        r[k] = frac * HALF_SPAN;
        area[k] = (1.15 - 0.3 * frac) * CHORD * HALF_SPAN / N_SEG as f32;
        sum += area[k];
    }
    for k in 0..N_SEG { area[k] *= CHORD * HALF_SPAN / sum; }
    (r, area)
}

fn surfaces(j: &[f32; 9], seg_r: &[f32; N_SEG], seg_a: &[f32; N_SEG]) -> Surf {
    let mut out = Surf::default();
    for side in 0..2 {
        let sign = if side == 0 { 1.0 } else { -1.0 };
        let (flap, sweep, twist) = (j[3 * side], j[3 * side + 1], j[3 * side + 2]);
        let r = mm(&mm(&rot_x(sign * flap), &rot_z(sign * sweep)), &rot_y(-(twist + WING_INCIDENCE)));
        let sh = [SHOULDER[0], sign * SHOULDER[1], SHOULDER[2]];
        let s = scale(col(&r, 1), sign);
        let c = col(&r, 0);
        let n = col(&r, 2);
        for k in 0..N_SEG {
            let i = side * N_SEG + k;
            out.r[i] = add(sh, scale(s, seg_r[k]));
            out.c[i] = c; out.n[i] = n; out.s[i] = s; out.area[i] = seg_a[k];
        }
    }
    let rt = mm(&rot_x(j[7]), &rot_y(-j[6]));
    let i = N_SURF - 1;
    out.r[i] = TAIL_POS; out.c[i] = col(&rt, 0); out.s[i] = col(&rt, 1); out.n[i] = col(&rt, 2);
    out.area[i] = TAIL_AREA.0 + (TAIL_AREA.1 - TAIL_AREA.0) * j[8];
    out
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

fn coeffs(alpha: f32) -> (f32, f32) {
    let sig = sigmoid((alpha.abs() - 0.26) / 0.04);
    let cl_lin = 2.0 * PI * alpha;
    let cl_post = (2.0 * alpha).sin();
    let cl = (1.0 - sig) * cl_lin + sig * cl_post;
    let sa = alpha.sin();
    let cd = 0.02 + (1.0 - sig) * 0.06 * cl_lin * cl_lin + sig * 1.8 * sa * sa;
    (cl, cd)
}

fn aero(v_air_b: V3, omega: V3, sf: &Surf, r_dot: &[V3; N_SURF]) -> (V3, V3) {
    let mut f = [0.0; 3];
    let mut tau = [0.0; 3];
    for i in 0..N_SURF {
        let mut u = add(add(v_air_b, cross(omega, sf.r[i])), r_dot[i]);
        u = sub(u, scale(sf.s[i], dot(u, sf.s[i])));
        let speed = norm(u).max(1e-4);
        let uh = scale(u, 1.0 / speed);
        let alpha = (-dot(u, sf.n[i])).atan2(dot(u, sf.c[i]));
        let (cl, cd) = coeffs(alpha);
        let q = 0.5 * RHO * speed * speed * sf.area[i];
        let mut ld = sub(sf.n[i], scale(uh, dot(sf.n[i], uh)));
        ld = scale(ld, 1.0 / norm(ld).max(1e-6));
        let fi = sub(scale(ld, q * cl), scale(uh, q * cd));
        f = add(f, fi);
        tau = add(tau, cross(sf.r[i], fi));
    }
    (f, tau)
}

// ------------------------------------------------------------------ sim
struct Sim {
    birds: Vec<Bird>,
    rng: Rng,
    policy: Option<Policy>,
    seg_r: [f32; N_SEG],
    seg_a: [f32; N_SEG],
    ring_radius: f32,
    wind: V3,
    turb: f32,
    gust: Vec<V3>,
    noise: f32,
    out: Vec<f32>,
    obs_out: Vec<f32>,
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    time: f32,
}

impl Sim {
    fn spawn_ring(&mut self, i: usize, from: V3, heading: f32) {
        let r = &mut self.rng;
        let yaw = heading + r.sym() * 70f32.to_radians();
        let dist = 18.0 + r.next() * 17.0;
        let dz = r.sym() * 7.0;
        let p = [from[0] + dist * yaw.cos(), from[1] + dist * yaw.sin(), (from[2] + dz).clamp(15.0, 70.0)];
        let mut d = sub(p, from);
        d[2] *= 0.5;
        let b = &mut self.birds[i];
        b.ring_pos = p;
        b.ring_n = scale(d, 1.0 / norm(d));
    }

    fn reset(&mut self, i: usize) {
        let r = &mut self.rng;
        let yaw = r.next() * 2.0 * PI;
        let z = 35.0 + r.next() * 10.0;
        let pitch = r.sym() * 0.15;
        let roll = r.sym() * 0.2;
        let speed = 8.0 + r.next() * 3.0;
        let qy = [(yaw / 2.).cos(), 0., 0., (yaw / 2.).sin()];
        let qp = [(pitch / 2.).cos(), 0., (pitch / 2.).sin(), 0.];
        let qr = [(roll / 2.).cos(), (roll / 2.).sin(), 0., 0.];
        let q = quat_mul(quat_mul(qy, qp), qr);
        let fwd = col(&quat_to_mat(q), 0);
        let total = self.birds[i].total_rings;
        let mut b = Bird::default();
        b.pos = [0.0, 0.0, z];
        b.quat = q;
        b.vel = scale(fwd, speed);
        b.joints[8] = 0.5;
        b.target = b.joints;
        b.total_rings = total;
        b.last_ring = b.pos;
        self.birds[i] = b;
        let h = b.vel[1].atan2(b.vel[0]);
        self.spawn_ring(i, b.pos, h);
    }

    fn obs(&self, b: &Bird) -> [f32; N_OBS] {
        let r = quat_to_mat(b.quat);
        let v_b = mtv(&r, b.vel);
        let g_b = [-r[2][0], -r[2][1], -r[2][2]];
        let to_ring = mtv(&r, sub(b.ring_pos, b.pos));
        let dist = norm(to_ring);
        let rn = mtv(&r, b.ring_n);
        let mut o = [0.0; N_OBS];
        let mut k = 0;
        let mut put = |x: f32| { o[k] = x; k += 1; };
        for a in 0..3 { put(v_b[a] / 10.0); }
        for a in 0..3 { put(b.omega[a] / 5.0); }
        for a in 0..3 { put(g_b[a]); }
        for a in 0..3 { put(to_ring[a] / dist.max(1e-3)); }
        put(dist / 20.0);
        for a in 0..3 { put(to_ring[a].clamp(-30.0, 30.0) / 20.0); }
        for a in 0..3 { put(rn[a]); }
        put(b.pos[2] / 40.0);
        for a in 0..9 { put((b.joints[a] - J_LO[a]) / (J_HI[a] - J_LO[a]) * 2.0 - 1.0); }
        for a in 0..9 { put((b.target[a] - J_LO[a]) / (J_HI[a] - J_LO[a]) * 2.0 - 1.0); }
        o
    }

    fn step_bird(&mut self, i: usize, action: [f32; N_ACT]) {
        let h = DT / SUBSTEPS as f32;
        let wind = add(self.wind, self.gust[i]);
        let mut b = self.birds[i];
        for a in 0..9 {
            b.target[a] = J_LO[a] + (action[a].clamp(-1.0, 1.0) + 1.0) * 0.5 * (J_HI[a] - J_LO[a]);
        }
        let start = b.pos;
        let mut r_prev = surfaces(&b.joints, &self.seg_r, &self.seg_a).r;
        for _ in 0..SUBSTEPS {
            for a in 0..9 {
                let lim = J_RATE[a] * h;
                b.joints[a] += (b.target[a] - b.joints[a]).clamp(-lim, lim);
            }
            let sf = surfaces(&b.joints, &self.seg_r, &self.seg_a);
            let mut r_dot = [[0.0; 3]; N_SURF];
            for k in 0..N_SURF { r_dot[k] = scale(sub(sf.r[k], r_prev[k]), 1.0 / h); }
            let rm = quat_to_mat(b.quat);
            let v_air_b = mtv(&rm, sub(b.vel, wind));
            let (mut f_b, tau) = aero(v_air_b, b.omega, &sf, &r_dot);
            f_b = sub(f_b, scale(v_air_b, 0.5 * RHO * BODY_CDA * norm(v_air_b)));
            let mut f_w = mv(&rm, f_b);
            f_w[2] -= MASS * G;
            b.vel = add(b.vel, scale(f_w, h / MASS));
            b.pos = add(b.pos, scale(b.vel, h));
            let iw = [INERTIA[0] * b.omega[0], INERTIA[1] * b.omega[1], INERTIA[2] * b.omega[2]];
            let gyro = cross(b.omega, iw);
            for a in 0..3 {
                b.omega[a] = (b.omega[a] + (tau[a] - gyro[a]) / INERTIA[a] * h).clamp(-30.0, 30.0);
            }
            let dq = quat_mul(b.quat, [0.0, b.omega[0], b.omega[1], b.omega[2]]);
            let mut q = [0.0; 4];
            for a in 0..4 { q[a] = b.quat[a] + 0.5 * h * dq[a]; }
            let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            for a in 0..4 { q[a] /= qn; }
            b.quat = q;
            r_prev = sf.r;
            b.speed_air = norm(v_air_b);
        }
        b.since_ring += 1;
        b.passed_flag = 0.0;

        let d0 = dot(sub(start, b.ring_pos), b.ring_n);
        let d1 = dot(sub(b.pos, b.ring_pos), b.ring_n);
        let mut passed = false;
        if d0 < 0.0 && d1 >= 0.0 {
            let t = d0 / (d0 - d1 + 1e-8);
            let hit = add(start, scale(sub(b.pos, start), t));
            passed = norm(sub(hit, b.ring_pos)) < self.ring_radius;
        }
        let bad = !(b.pos[0].is_finite() && b.pos[1].is_finite() && b.pos[2].is_finite());
        let crash = b.pos[2] < 0.0 || b.pos[2] > 150.0 || bad;
        self.birds[i] = b;
        if passed {
            let b = &mut self.birds[i];
            b.rings += 1; b.total_rings += 1; b.since_ring = 0; b.passed_flag = 1.0;
            b.last_ring = b.ring_pos;
            let from = b.ring_pos;
            let heading = b.vel[1].atan2(b.vel[0]);
            self.spawn_ring(i, from, heading);
        }
        if bad {
            self.reset(i);
        } else if crash {
            let b = &mut self.birds[i];
            b.crashed = 1e-3;
            b.pos[2] = b.pos[2].max(0.05);
        } else if self.birds[i].since_ring > RING_TIMEOUT {
            self.reset(i);
        }
    }

    fn step(&mut self) {
        self.time += DT;
        // turbulence: per-bird Ornstein-Uhlenbeck gusts
        for i in 0..self.birds.len() {
            for a in 0..3 {
                let g = &mut self.gust[i][a];
                *g += -*g * DT / 1.5 + self.turb * (2.0 * DT / 1.5).sqrt() * self.rng.gauss() * if a == 2 { 0.4 } else { 1.0 };
            }
        }
        for i in 0..self.birds.len() {
            if self.birds[i].crashed > 0.0 {
                self.birds[i].crashed += DT;
                if self.birds[i].crashed > 1.5 { self.reset(i); }
                continue;
            }
            let o = self.obs(&self.birds[i]);
            let mut a = [0.0; N_ACT];
            if let Some(p) = &self.policy {
                a = p.act(&o, &mut self.buf_a, &mut self.buf_b);
                if self.noise > 0.0 {
                    for k in 0..N_ACT { a[k] += self.noise * p.log_std[k].exp() * self.rng.gauss(); }
                }
            }
            self.step_bird(i, a);
        }
    }

    fn write_out(&mut self) {
        let n = self.birds.len();
        self.out.resize(n * STRIDE, 0.0);
        for (i, b) in self.birds.iter().enumerate() {
            let o = &mut self.out[i * STRIDE..(i + 1) * STRIDE];
            o[0..3].copy_from_slice(&b.pos);
            o[3..7].copy_from_slice(&b.quat);
            o[7..16].copy_from_slice(&b.joints);
            o[16..19].copy_from_slice(&b.ring_pos);
            o[19..22].copy_from_slice(&b.ring_n);
            o[22..25].copy_from_slice(&b.vel);
            o[25] = b.rings as f32;
            o[26] = b.crashed;
            o[27] = b.total_rings as f32;
            o[28] = b.passed_flag;
            o[29] = b.speed_air;
            o[30] = b.since_ring as f32 * DT;
            o[31] = 0.0;
        }
    }
}

static mut SIM: Option<Sim> = None;

fn sim() -> &'static mut Sim {
    unsafe {
        if SIM.is_none() {
            let (seg_r, seg_a) = seg_geom();
            SIM = Some(Sim {
                birds: vec![], rng: Rng(0x9E3779B97F4A7C15), policy: None, seg_r, seg_a,
                ring_radius: 2.0, wind: [0.0; 3], turb: 0.0, gust: vec![], noise: 0.0,
                out: vec![], obs_out: vec![0.0; N_OBS], buf_a: Vec::with_capacity(512), buf_b: Vec::with_capacity(512), time: 0.0,
            });
        }
        SIM.as_mut().unwrap()
    }
}

// ------------------------------------------------------------------ exports
/// Allocate `n` floats in wasm memory (for JS to write weights into).
#[no_mangle]
pub extern "C" fn alloc_f32(n: usize) -> *mut f32 {
    let mut v = vec![0.0f32; n];
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// Layout: [n_layers, d0..dn, mean(d0), var(d0), log_std(dn), (W row-major [out,in], b[out]) per layer]
#[no_mangle]
pub extern "C" fn load_policy(ptr: *const f32, n: usize) -> i32 {
    let d = unsafe { std::slice::from_raw_parts(ptr, n) };
    let nl = d[0] as usize;
    let dims: Vec<usize> = (0..=nl).map(|i| d[1 + i] as usize).collect();
    if dims[0] != N_OBS || dims[nl] != N_ACT { return -1; }
    let mut k = 2 + nl;
    let mut take = |m: usize| { let s = d[k..k + m].to_vec(); k += m; s };
    let mean = take(N_OBS);
    let var = take(N_OBS);
    let log_std = take(N_ACT);
    let mut layers = vec![];
    for l in 0..nl {
        let (i, o) = (dims[l], dims[l + 1]);
        let w = take(i * o);
        let b = take(o);
        layers.push(Layer { w, b, n_in: i, n_out: o });
    }
    let std = var.iter().map(|v| v.sqrt()).collect();
    sim().policy = Some(Policy { mean, std, layers, log_std });
    0
}

#[no_mangle]
pub extern "C" fn init(n: usize, seed: u32) {
    let s = sim();
    s.rng = Rng(0x9E3779B97F4A7C15 ^ ((seed as u64) << 17) ^ seed as u64);
    s.birds = vec![Bird::default(); n];
    s.gust = vec![[0.0; 3]; n];
    for i in 0..n { s.reset(i); }
    s.write_out();
}

#[no_mangle]
pub extern "C" fn step(k: usize) {
    let s = sim();
    for _ in 0..k { s.step(); }
    s.write_out();
}

#[no_mangle] pub extern "C" fn reset_bird(i: usize) { let s = sim(); if i < s.birds.len() { s.reset(i); s.write_out(); } }
#[no_mangle] pub extern "C" fn state_ptr() -> *const f32 { sim().out.as_ptr() }
#[no_mangle] pub extern "C" fn stride() -> usize { STRIDE }
#[no_mangle] pub extern "C" fn dt() -> f32 { DT }
#[no_mangle] pub extern "C" fn set_ring_radius(r: f32) { sim().ring_radius = r; }
#[no_mangle] pub extern "C" fn set_wind(x: f32, y: f32, z: f32, turb: f32) { let s = sim(); s.wind = [x, y, z]; s.turb = turb; }
#[no_mangle] pub extern "C" fn set_noise(n: f32) { sim().noise = n; }

/// Debug/verification: overwrite bird i with [pos3 vel3 quat4 omega3 joints9 ring_pos3 ring_n3] (28 floats)
#[no_mangle]
pub extern "C" fn debug_set(i: usize, ptr: *const f32) {
    let d = unsafe { std::slice::from_raw_parts(ptr, 28) };
    let s = sim();
    let b = &mut s.birds[i];
    b.pos.copy_from_slice(&d[0..3]); b.vel.copy_from_slice(&d[3..6]);
    b.quat.copy_from_slice(&d[6..10]); b.omega.copy_from_slice(&d[10..13]);
    b.joints.copy_from_slice(&d[13..22]); b.target = b.joints;
    b.ring_pos.copy_from_slice(&d[22..25]); b.ring_n.copy_from_slice(&d[25..28]);
    b.since_ring = 0; b.crashed = 0.0;
    s.write_out();
}

/// Debug: observation vector of bird i (to compare against Python).
#[no_mangle]
pub extern "C" fn debug_obs(i: usize) -> *const f32 {
    let s = sim();
    let o = s.obs(&s.birds[i]);
    s.obs_out.copy_from_slice(&o);
    s.obs_out.as_ptr()
}
