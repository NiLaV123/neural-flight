"""
Batched 6-DOF flapping-bird flight simulator, written entirely in PyTorch so that
thousands of birds are simulated in parallel on the GPU.

Frames: world = (x east, y north, z up). Body = (x forward, y left, z up).

Physics:
  * rigid body: position, velocity, orientation quaternion, body angular velocity
  * gravity, Euler's rotation equations with an inertia tensor
  * blade-element aerodynamics: each wing is split into spanwise segments, the
    tail is one surface. For every surface the local airflow includes body
    velocity, rotation (w x r) AND the surface's own motion from the joints, so
    a downstroke really pushes air and produces lift/thrust.
  * lift/drag from angle of attack with a stall model, body drag
  * wing momentum (default): each wing has mass spread along its span. Flap and
    sweep are driven by torque-limited muscles (PD servos) against the wing's
    inertia, aerodynamic load and gravity, and the wing's momentum pushes back on
    the body: a hard downstroke makes the body recoil, and the muscles cost
    energy. Twist and tail joints stay as light, rate-limited servos.
    With wing_dynamics=False the original massless, rate-limited wings are used
    (this is what older checkpoints were trained with).

Species: see SPECIES below. Each has its own size, mass, wing shape, muscle
strength and flight speed, and needs its own trained policy.

Controls (9 per bird, each in [-1, 1]):
  left wing:  flap (dihedral), sweep (fore/aft), twist (incidence)
  right wing: flap, sweep, twist
  tail:       pitch (elevator), tilt (roll of tail -> yaw/side force), spread (area)
"""
import math
import torch

RHO = 1.225
G = 9.81


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def rot_x(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(a.shape + (3, 3))


def rot_y(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(a.shape + (3, 3))


def rot_z(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(a.shape + (3, 3))


def mv(m, v):  # batched matrix @ vector
    return (m @ v.unsqueeze(-1)).squeeze(-1)


def _species(mass, half_span, chord, speed, ring_min_r, substeps=4, wing_frac=0.07, flap_strength=2.0,
             tail_frac=0.63, colors=("#3b4a63", "#d9dde6", "#2d3a50")):
    """Build a species from a few measurements; everything else scales from the
    original 0.8 kg / 1.2 m-span bird ("hawk")."""
    L = half_span / 0.6
    k_inertia = (mass / 0.8) * L ** 2
    return dict(
        MASS=mass, HALF_SPAN=half_span, CHORD=chord,
        INERTIA=tuple(i * k_inertia for i in (0.030, 0.035, 0.060)),
        SHOULDER=(0.0, 0.06 * L, 0.02 * L), TAIL_POS=(-0.38 * L, 0.0, 0.0),
        TAIL_AREA=tuple(a * (chord / 0.20) * L * tail_frac / 0.63 for a in (0.02, 0.06)),
        BODY_CDA=0.006 * (mass / 0.8) ** (2 / 3),
        START_SPEED=speed, RING_MIN_R=ring_min_r, SUBSTEPS=substeps,
        WING_MASS=wing_frac * mass, FLAP_STRENGTH=flap_strength, SCALE=L, COLORS=colors,
    )


# name: (mass kg, half span m, chord m, launch speed range m/s, smallest ring radius m, ...)
SPECIES = {
    "sparrow":   _species(0.03, 0.12, 0.065, (6.0, 8.0), 1.0, substeps=8, flap_strength=2.6,
                          colors=("#7a5a3a", "#c9a882", "#5a4030")),
    "pigeon":    _species(0.35, 0.33, 0.11, (8.5, 11.5), 1.5, substeps=6, colors=("#6b7280", "#c4c9d4", "#4b5563")),
    "hawk":      _species(0.8, 0.6, 0.20, (8.0, 11.0), 2.0),
    "eagle":     _species(4.5, 1.05, 0.33, (10.5, 13.5), 2.5, flap_strength=1.8, colors=("#4a3423", "#8a6a4a", "#f2efe8")),
    "albatross": _species(8.5, 1.5, 0.18, (14.0, 18.0), 3.0, flap_strength=1.4, tail_frac=0.5,
                          colors=("#f4f4f2", "#33302c", "#e8e8e6")),
}
SPECIES_INFO = {
    "sparrow": "Tiny, fast-flapping songbird (30 g, 24 cm span)",
    "pigeon": "Agile all-rounder (350 g, 66 cm span)",
    "hawk": "The original bird (800 g, 1.2 m span)",
    "eagle": "Big, broad wings, slow powerful beats (4.5 kg, 2.1 m span)",
    "albatross": "Long, thin soaring wings, needs speed (8.5 kg, 3 m span)",
}


class BirdEnv:
    # ---- bird parameters ----
    MASS = 0.8
    INERTIA = (0.030, 0.035, 0.060)          # roll, pitch, yaw  (kg m^2)
    N_SEG = 3                                # blade elements per wing
    HALF_SPAN = 0.6
    CHORD = 0.20
    SHOULDER = (0.0, 0.06, 0.02)             # left shoulder; right is mirrored
    TAIL_POS = (-0.38, 0.0, 0.0)
    TAIL_AREA = (0.02, 0.06)                 # min/max spread
    BODY_CDA = 0.006

    # joint ranges (rad): (min, max)
    FLAP = (-0.7, 1.0)
    SWEEP = (-0.45, 0.45)
    TWIST = (-0.6, 0.6)
    TAIL_PITCH = (-0.7, 0.7)
    TAIL_TILT = (-0.8, 0.8)
    TAIL_SPREAD = (0.0, 1.0)
    # max joint speeds (rad/s) - servos
    JOINT_RATE = (18.0, 8.0, 12.0, 18.0, 8.0, 12.0, 8.0, 8.0, 6.0)

    WING_INCIDENCE = 0.06
    START_SPEED = (8.0, 11.0)                # launch speed range (m/s)
    RING_MIN_R = 2.0
    WING_MASS = 0.056                        # per wing (kg)
    FLAP_STRENGTH = 2.0                      # max flap torque, in units of m*g*half_span
    SCALE = 1.0
    COLORS = ("#3b4a63", "#d9dde6", "#2d3a50")
    DYN = (0, 1, 3, 4)                       # joints with real dynamics: flap & sweep of each wing
    DYN_DAMP = 0.7                           # servo damping ratio
    DYN_FREQ = (55.0, 40.0)                  # servo stiffness (rad/s natural freq, unloaded) flap, sweep
    QD_MAX = 80.0                            # safety cap on joint speed (rad/s)

    # ---- task parameters ----
    DT = 0.02                 # control step (50 Hz)
    SUBSTEPS = 4              # physics at 200 Hz
    MAX_STEPS = 2000          # 40 s episodes
    RING_TIMEOUT = 600        # 12 s to reach each ring
    N_OBS = 42                # 38 with wing_dynamics=False
    N_ACT = 9

    @staticmethod
    def n_obs(wing_dynamics=True):
        return 42 if wing_dynamics else 38

    def __init__(self, num_envs, device, ring_radius=None, compile=False, species="hawk", wing_dynamics=True,
                 energy_cost=0.0):
        if species not in SPECIES:
            raise ValueError(f"unknown species {species!r}; choose from {', '.join(SPECIES)}")
        for k, v in SPECIES[species].items():
            setattr(self, k, v)
        self.species = species
        self.wing_dynamics = wing_dynamics
        self.energy_cost = energy_cost
        self.N_OBS = self.n_obs(wing_dynamics)
        if ring_radius is None:
            ring_radius = 2 * self.RING_MIN_R
        self.n = num_envs
        self.dev = device
        self.ring_radius = ring_radius
        f = dict(device=device, dtype=torch.float32)
        self.f = f
        self.I = torch.tensor(self.INERTIA, **f)
        lo = [self.FLAP[0], self.SWEEP[0], self.TWIST[0]] * 2 + [self.TAIL_PITCH[0], self.TAIL_TILT[0], self.TAIL_SPREAD[0]]
        hi = [self.FLAP[1], self.SWEEP[1], self.TWIST[1]] * 2 + [self.TAIL_PITCH[1], self.TAIL_TILT[1], self.TAIL_SPREAD[1]]
        self.j_lo = torch.tensor(lo, **f)
        self.j_hi = torch.tensor(hi, **f)
        self.j_rate = torch.tensor(self.JOINT_RATE, **f)
        # segment centres along span (fraction of half span) and per-segment chord taper
        frac = (torch.arange(self.N_SEG, **f) + 0.5) / self.N_SEG
        self.seg_r = frac * self.HALF_SPAN
        taper = 1.15 - 0.3 * frac
        self.seg_area = taper * self.CHORD * self.HALF_SPAN / self.N_SEG
        self.seg_area = self.seg_area * (self.CHORD * self.HALF_SPAN / self.seg_area.sum())
        self.shoulder = torch.tensor(self.SHOULDER, **f)
        self.tail_pos = torch.tensor(self.TAIL_POS, **f)
        self.ex = torch.tensor([1.0, 0, 0], **f)
        self.ey = torch.tensor([0.0, 1, 0], **f)
        self.ez = torch.tensor([0.0, 0, 1], **f)
        self.mirror = torch.tensor([1.0, -1.0, 1.0], **f)
        self.down = torch.tensor([0.0, 0, -1], **f)

        # ---- wing mass & muscles (flap, sweep of each wing)
        self.seg_mass = self.WING_MASS * self.seg_area / self.seg_area.sum()          # [SEG]
        J = (self.seg_mass * self.seg_r ** 2).sum() + 1e-6
        self.J = J
        mgL = self.MASS * G * self.HALF_SPAN
        wf, ws = self.DYN_FREQ
        self.kp = torch.tensor([J * wf ** 2, J * ws ** 2] * 2, **f)
        self.kd = torch.tensor([2 * self.DYN_DAMP * J * wf, 2 * self.DYN_DAMP * J * ws] * 2, **f)
        self.tau_max = torch.tensor([self.FLAP_STRENGTH * mgL, 0.35 * self.FLAP_STRENGTH * mgL] * 2, **f)
        self.dyn_idx = torch.tensor(self.DYN, device=device)
        kin = torch.ones(9, dtype=torch.bool, device=device)
        if wing_dynamics:
            kin[self.dyn_idx] = False
        self.kin_mask = kin.to(torch.float32)

        z = lambda *s: torch.zeros((num_envs,) + s, **f)
        self.pos, self.vel = z(3), z(3)
        self.quat = z(4)
        self.omega = z(3)             # body frame
        self.joints = z(9)
        self.target = z(9)
        self.qd = z(4)                # flap/sweep joint velocities
        self.rdot = z(2 * self.N_SEG + 1, 3)   # surface velocities relative to body (for wing momentum)
        self.work = z()               # muscle work done in the last control step (J)
        self.ring_pos, self.ring_n = z(3), z(3)
        self.prev_dist = z()
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.since_ring = torch.zeros_like(self.steps)
        self.rings = torch.zeros_like(self.steps)
        impl = self._substep_dyn if wing_dynamics else self._substep_impl
        self._substep = torch.compile(impl, dynamic=False) if compile else impl
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    # ------------------------------------------------------------------ geometry
    def surfaces(self, joints, axes=False):
        """Return per-surface (position r, chord dir c, normal n, span dir s, area)
        in body frame. Shape [N, S, 3] with S = 2*N_SEG + 1.
        With axes=True also return the flap and sweep hinge axes, [N, 2, 3] each (left, right)."""
        n = joints.shape[0]
        out_r, out_c, out_n, out_s, out_a = [], [], [], [], []
        ax_flap, ax_sweep = [], []
        for side, sign in ((0, 1.0), (1, -1.0)):
            flap, sweep, twist = joints[:, 3 * side], joints[:, 3 * side + 1], joints[:, 3 * side + 2]
            # twist about span (y) axis: positive = leading edge up
            R = rot_x(sign * flap) @ rot_z(sign * sweep) @ rot_y(-(twist + self.WING_INCIDENCE))
            sh = self.shoulder if sign > 0 else self.shoulder * self.mirror
            s = R[..., :, 1] * sign
            c = R[..., :, 0]
            nn = R[..., :, 2]
            r = sh + s.unsqueeze(1) * self.seg_r.view(1, -1, 1)          # [N, SEG, 3]
            out_r.append(r)
            out_c.append(c.unsqueeze(1).expand(-1, self.N_SEG, -1))
            out_n.append(nn.unsqueeze(1).expand(-1, self.N_SEG, -1))
            out_s.append(s.unsqueeze(1).expand(-1, self.N_SEG, -1))
            out_a.append(self.seg_area.expand(n, -1))
            if axes:
                ax_flap.append((self.ex * sign).expand(n, 3))
                ax_sweep.append(rot_x(sign * flap)[..., :, 2] * sign)
        tp, tt, ts = joints[:, 6], joints[:, 7], joints[:, 8]
        Rt = rot_x(tt) @ rot_y(-tp)
        c, s, nn = Rt[..., :, 0], Rt[..., :, 1], Rt[..., :, 2]
        out_r.append(self.tail_pos.expand(n, 1, 3))
        out_c.append(c.unsqueeze(1)); out_n.append(nn.unsqueeze(1)); out_s.append(s.unsqueeze(1))
        out_a.append((self.TAIL_AREA[0] + (self.TAIL_AREA[1] - self.TAIL_AREA[0]) * ts).unsqueeze(1))
        out = (torch.cat(out_r, 1), torch.cat(out_c, 1), torch.cat(out_n, 1),
               torch.cat(out_s, 1), torch.cat(out_a, 1))
        if axes:
            out = out + (torch.stack(ax_flap, 1), torch.stack(ax_sweep, 1))
        return out

    # ------------------------------------------------------------------ aero
    @staticmethod
    def coeffs(alpha):
        a_stall = 0.26
        sig = torch.sigmoid((alpha.abs() - a_stall) / 0.04)
        cl_lin = 2 * math.pi * alpha
        cl_post = 1.0 * torch.sin(2 * alpha)
        cl = (1 - sig) * cl_lin + sig * cl_post
        cd = 0.02 + (1 - sig) * 0.06 * cl_lin ** 2 + sig * 1.8 * torch.sin(alpha) ** 2
        return cl, cd

    def aero(self, vel_b, omega, r, c, nrm, s, area, r_dot, per_surface=False):
        # velocity of each surface through the air, body frame
        u = vel_b.unsqueeze(1) + torch.cross(omega.unsqueeze(1).expand_as(r), r, dim=-1) + r_dot
        u = u - (u * s).sum(-1, keepdim=True) * s          # drop spanwise flow
        speed = u.norm(dim=-1).clamp_min(1e-4)
        uh = u / speed.unsqueeze(-1)
        alpha = torch.atan2(-(u * nrm).sum(-1), (u * c).sum(-1))
        cl, cd = self.coeffs(alpha)
        q = 0.5 * RHO * speed ** 2 * area
        lift_dir = nrm - (nrm * uh).sum(-1, keepdim=True) * uh
        lift_dir = lift_dir / lift_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        F = (q * cl).unsqueeze(-1) * lift_dir - (q * cd).unsqueeze(-1) * uh
        torque = torch.cross(r, F, dim=-1).sum(1)
        if per_surface:
            return F
        return F.sum(1), torque

    # ------------------------------------------------------------------ rings
    def spawn_ring(self, mask, from_pos, heading):
        k = int(mask.sum())
        if k == 0:
            return
        yaw = heading[mask] + (torch.rand(k, **self.f) * 2 - 1) * math.radians(70)
        dist = (18 + torch.rand(k, **self.f) * 17) * max(1.0, sum(self.START_SPEED) / 20)
        dz = (torch.rand(k, **self.f) * 2 - 1) * 7
        p = from_pos[mask].clone()
        p[:, 0] += dist * torch.cos(yaw)
        p[:, 1] += dist * torch.sin(yaw)
        p[:, 2] = (p[:, 2] + dz).clamp(15, 70)
        d = p - from_pos[mask]
        d[:, 2] *= 0.5
        self.ring_pos[mask] = p
        self.ring_n[mask] = d / d.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ reset
    def reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        f = self.f
        yaw = torch.rand(k, **f) * 2 * math.pi
        self.pos[mask] = torch.stack([torch.zeros(k, **f), torch.zeros(k, **f), 35 + torch.rand(k, **f) * 10], -1)
        pitch = (torch.rand(k, **f) * 2 - 1) * 0.15
        roll = (torch.rand(k, **f) * 2 - 1) * 0.2
        qy = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
        qp = torch.stack([torch.cos(pitch / 2), 0 * yaw, torch.sin(pitch / 2), 0 * yaw], -1)
        qr = torch.stack([torch.cos(roll / 2), torch.sin(roll / 2), 0 * yaw, 0 * yaw], -1)
        q = quat_mul(quat_mul(qy, qp), qr)
        self.quat[mask] = q
        lo, hi = self.START_SPEED
        speed = lo + torch.rand(k, **f) * (hi - lo)
        fwd = quat_to_mat(q)[:, :, 0]
        self.vel[mask] = fwd * speed.unsqueeze(-1)
        self.omega[mask] = 0
        j = torch.zeros(k, 9, **f)
        j[:, 8] = 0.5
        self.joints[mask] = j
        self.target[mask] = j
        self.qd[mask] = 0
        self.rdot[mask] = 0
        self.steps[mask] = 0
        self.since_ring[mask] = 0
        self.rings[mask] = 0
        self.spawn_ring(mask, self.pos, torch.atan2(self.vel[:, 1], self.vel[:, 0]))
        self.prev_dist[mask] = (self.ring_pos[mask] - self.pos[mask]).norm(dim=-1)

    # ------------------------------------------------------------------ obs
    def obs(self):
        R = quat_to_mat(self.quat)
        Rt = R.transpose(-1, -2)
        v_b = mv(Rt, self.vel)
        g_b = -R[:, 2, :]
        to_ring = mv(Rt, self.ring_pos - self.pos)
        dist = to_ring.norm(dim=-1, keepdim=True)
        ring_n_b = mv(Rt, self.ring_n)
        jn = (self.joints - self.j_lo) / (self.j_hi - self.j_lo) * 2 - 1
        tn = (self.target - self.j_lo) / (self.j_hi - self.j_lo) * 2 - 1
        parts = [
            v_b / 10, self.omega / 5, g_b,
            to_ring / dist.clamp_min(1e-3), dist / 20, to_ring.clamp(-30, 30) / 20,
            ring_n_b, self.pos[:, 2:3] / 40,
            jn, tn,
        ]
        if self.wing_dynamics:
            parts.append(self.qd / 30)
        return torch.cat(parts, -1)

    # ------------------------------------------------------------------ physics
    def _substep_impl(self, pos, vel, quat, omega, joints, target, r_prev):
        h = self.DT / self.SUBSTEPS
        Iv = self.I
        dj = (target - joints).clamp(-self.j_rate * h, self.j_rate * h)
        joints = joints + dj
        r, c, nrm, s, area = self.surfaces(joints)
        r_dot = (r - r_prev) / h
        R = quat_to_mat(quat)
        v_b = mv(R.transpose(-1, -2), vel)
        F_b, tau = self.aero(v_b, omega, r, c, nrm, s, area, r_dot)
        F_b = F_b - 0.5 * RHO * self.BODY_CDA * v_b.norm(dim=-1, keepdim=True) * v_b
        F_w = mv(R, F_b) + self.down * (self.MASS * G)
        vel = vel + F_w / self.MASS * h
        pos = pos + vel * h
        Iw = Iv * omega
        dom = (tau - torch.cross(omega, Iw, dim=-1)) / Iv
        omega = (omega + dom * h).clamp(-30, 30)
        dq = quat_mul(quat, torch.cat([torch.zeros_like(omega[:, :1]), omega], -1))
        quat = quat + 0.5 * h * dq
        quat = quat / quat.norm(dim=-1, keepdim=True)
        return pos, vel, quat, omega, joints, r

    def _substep_dyn(self, pos, vel, quat, omega, joints, target, qd, rdot_prev, work):
        """One physics substep with massive wings (see module docstring)."""
        h = self.DT / self.SUBSTEPS
        S = self.N_SEG
        R = quat_to_mat(quat)
        Rt = R.transpose(-1, -2)
        v_b = mv(Rt, vel)
        g_b = mv(Rt, self.down) * G

        # light joints (twist, tail): rate-limited servos as before
        dj = (target - joints).clamp(-self.j_rate * h, self.j_rate * h) * self.kin_mask
        joints = joints + dj

        r, c, nrm, s, area, axf, axs = self.surfaces(joints, axes=True)
        sh = torch.stack([self.shoulder, self.shoulder * self.mirror])                 # [2, 3]
        d = r[:, :2 * S].reshape(-1, 2, S, 3) - sh.view(1, 2, 1, 3)                       # wing segments rel. shoulder
        lever_f = torch.cross(axf.unsqueeze(2).expand_as(d), d, dim=-1)                   # dr/d(flap)
        lever_s = torch.cross(axs.unsqueeze(2).expand_as(d), d, dim=-1)                   # dr/d(sweep)
        q4 = joints[:, self.dyn_idx]
        qf, qs = qd[:, 0::2], qd[:, 1::2]                                                 # [N, 2] (left, right)
        rdot_w = lever_f * qf[..., None, None] + lever_s * qs[..., None, None]           # [N, 2, S, 3]
        r_dot = torch.cat([rdot_w.reshape(-1, 2 * S, 3), torch.zeros_like(r[:, 2 * S:])], 1)

        F = self.aero(v_b, omega, r, c, nrm, s, area, r_dot, per_surface=True)           # [N, SURF, 3]
        # generalised forces on flap/sweep from air and wing weight
        Fw = F[:, :2 * S].reshape(-1, 2, S, 3) + self.seg_mass.view(1, 1, S, 1) * g_b.view(-1, 1, 1, 3)
        Q = torch.stack([(lever_f * Fw).sum((-1, -2)), (lever_s * Fw).sum((-1, -2))], -1).reshape(-1, 4)
        J4 = (self.seg_mass.view(1, 1, S, 1) * torch.stack([lever_f, lever_s], 1) ** 2).sum((-1, -2))  # [N,2,2]
        J4 = J4.permute(0, 2, 1).reshape(-1, 4) + 1e-7
        tgt4 = target[:, self.dyn_idx]
        tau = (self.kp * (tgt4 - q4) - self.kd * qd).clamp(-self.tau_max, self.tau_max)
        qd = (qd + (tau + Q) / J4 * h).clamp(-self.QD_MAX, self.QD_MAX)
        q4n = q4 + qd * h
        lo, hi = self.j_lo[self.dyn_idx], self.j_hi[self.dyn_idx]
        hit = ((q4n <= lo) & (qd < 0)) | ((q4n >= hi) & (qd > 0))
        qd = torch.where(hit, torch.zeros_like(qd), qd)
        q4n = torch.minimum(torch.maximum(q4n, lo), hi)
        joints = joints.index_copy(1, self.dyn_idx, q4n)
        work = work + (tau * qd).abs().sum(-1) * h

        # body: external forces minus what went into accelerating the wing masses
        rddot = (r_dot - rdot_prev) / h
        m_s = torch.cat([self.seg_mass, self.seg_mass, torch.zeros_like(self.seg_mass[:1])]).view(1, -1, 1)
        F_in = (m_s * rddot).sum(1)
        tau_in = torch.cross(r, m_s * rddot, dim=-1).sum(1)
        F_b = F.sum(1) - F_in - 0.5 * RHO * self.BODY_CDA * v_b.norm(dim=-1, keepdim=True) * v_b
        tau_b = torch.cross(r, F, dim=-1).sum(1) - tau_in
        F_w = mv(R, F_b) + self.down * (self.MASS * G)
        vel = vel + F_w / self.MASS * h
        pos = pos + vel * h
        Iw = self.I * omega
        dom = (tau_b - torch.cross(omega, Iw, dim=-1)) / self.I
        omega = (omega + dom * h).clamp(-30, 30)
        dq = quat_mul(quat, torch.cat([torch.zeros_like(omega[:, :1]), omega], -1))
        quat = quat + 0.5 * h * dq
        quat = quat / quat.norm(dim=-1, keepdim=True)
        return pos, vel, quat, omega, joints, qd, r_dot, work

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = action.clamp(-1, 1)
        self.target = self.j_lo + (action + 1) * 0.5 * (self.j_hi - self.j_lo)
        start_pos = self.pos.clone()
        if self.wing_dynamics:
            work = torch.zeros_like(self.work)
            for _ in range(self.SUBSTEPS):
                (self.pos, self.vel, self.quat, self.omega, self.joints, self.qd, self.rdot, work) = self._substep(
                    self.pos, self.vel, self.quat, self.omega, self.joints, self.target, self.qd, self.rdot, work)
            self.work = work
        else:
            r_prev = self.surfaces(self.joints)[0]
            for _ in range(self.SUBSTEPS):
                (self.pos, self.vel, self.quat, self.omega, self.joints, r_prev) = self._substep(
                    self.pos, self.vel, self.quat, self.omega, self.joints, self.target, r_prev)

        self.steps += 1
        self.since_ring += 1

        # ---- ring crossing: segment start->end crosses ring plane inside radius
        d0 = ((start_pos - self.ring_pos) * self.ring_n).sum(-1)
        d1 = ((self.pos - self.ring_pos) * self.ring_n).sum(-1)
        crossed = (d0 < 0) & (d1 >= 0)
        t = d0 / (d0 - d1 + 1e-8)
        hit_pt = start_pos + (self.pos - start_pos) * t.unsqueeze(-1)
        passed = crossed & ((hit_pt - self.ring_pos).norm(dim=-1) < self.ring_radius)

        dist = (self.ring_pos - self.pos).norm(dim=-1)
        reward = (self.prev_dist - dist).clamp(-1, 1)           # progress toward ring (m)
        reward = reward + 10.0 * passed.float()
        reward = reward - 0.002 * (action ** 2).sum(-1)
        if self.energy_cost and self.wing_dynamics:
            reward = reward - self.energy_cost * self.work / self.MASS          # J per kg of bird

        crash = (self.pos[:, 2] < 0) | ~torch.isfinite(self.vel).all(-1) | (self.pos[:, 2] > 150) | ~torch.isfinite(self.pos).all(-1)
        lost = self.since_ring > self.RING_TIMEOUT
        reward = reward - 5.0 * crash.float()

        self.rings += passed.long()
        self.since_ring[passed] = 0
        heading = torch.atan2(self.vel[:, 1], self.vel[:, 0])
        self.spawn_ring(passed, self.ring_pos.clone(), heading)
        self.prev_dist = (self.ring_pos - self.pos).norm(dim=-1)

        terminated = crash | lost
        truncated = (self.steps >= self.MAX_STEPS) & ~terminated
        info = {"passed": passed, "rings": self.rings.clone(), "done": terminated | truncated,
                "power": self.work / self.DT if self.wing_dynamics else torch.zeros_like(reward)}
        return reward, terminated, truncated, info
