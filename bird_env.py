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
  * wing physics, three versions (checkpoints remember which one they used):
      "wings-v2" (default): each wing has mass spread along its span and chord.
        Flap, sweep and twist are driven by torque-limited muscles (PD servos)
        against the wing's inertia, aerodynamic load (lift acts behind the twist
        axis, so the air twists the wing), gravity and the inertial forces of a
        moving, rotating body (centrifugal, Coriolis, body acceleration). The body
        sees the wings' full effect: a time-varying inertia tensor, recoil from
        the wing masses' acceleration, and gyroscopic/Coriolis coupling.
      "wings-v1": flap and sweep have mass and momentum; twist is a servo and
        rotating-frame terms are ignored.
      "massless": the original model; every joint is an ideal rate-limited servo.
  * energy budget (option): muscle work and a resting metabolism drain a reserve;
    each ring passed is food. An exhausted bird's muscles are weak, so it must glide.
  * soaring (option): wind with a realistic boundary-layer profile (stronger
    higher up), rising thermals and gentle sink between them. Birds feel the wind
    in their airspeed and can sense the nearest thermal.

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


def skew(r):
    """Cross-product matrix: skew(r) @ v == r x v.  r [..., 3] -> [..., 3, 3]."""
    x, y, z = r.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(r.shape + (3,))


def spd_inverse(M):
    """Inverse of a batch of small symmetric positive-definite matrices [n, k, k] by Gauss-Jordan
    elimination without pivoting (safe for SPD). Plain tensor ops, so torch.compile fuses it on any
    GPU instead of falling back to a generic linear-algebra kernel."""
    n, k, _ = M.shape
    eye = torch.eye(k, dtype=M.dtype, device=M.device)
    A = torch.cat([M, eye.expand(n, k, k)], -1)
    for i in range(k):
        piv = A[:, i:i + 1, :] / A[:, i:i + 1, i:i + 1]
        A = A - (A[:, :, i:i + 1] - eye[:, i:i + 1]) * piv
    return A[:, :, k:]


def mv(m, v):  # batched matrix @ vector
    return (m @ v.unsqueeze(-1)).squeeze(-1)


def _species(mass, half_span, chord, speed, ring_min_r, substeps=4, wing_frac=0.07, flap_strength=2.0,
             tail_frac=0.63, colors=("#3b4a63", "#d9dde6", "#2d3a50"), energy_per_kg=2500.0):
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
        ENERGY_PER_KG=energy_per_kg,
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
    QD_MAX = 80.0                            # cap on joint speed (rad/s); wings-v2 uses QD_MAX_V2
    QD_MAX_V2 = 150.0                        # wings-v2: safety cap only, rarely reached
    # wings-v2: chordwise layout relative to the twist axis (fractions of chord, + = forward)
    X_AERO = -0.05                           # aerodynamic centre sits just behind the twist axis
    X_MASS = -0.15                           # wing mass centroid
    TWIST_FREQ = 70.0
    TWIST_STRENGTH = 0.15                    # max twist torque, units of m*g*chord (x FLAP_STRENGTH)
    # energy budget
    ENERGY_PER_KG = 2500.0                   # reserve (J/kg)
    BASAL_W_PER_KG = 3.0                     # resting metabolism
    RING_FOOD = 0.2                          # fraction of the reserve restored per ring
    WEAK = 0.1                               # muscle strength when exhausted
    # soaring
    N_THERMALS = 6
    Z0 = 0.03                                # surface roughness (m) for the wind profile
    WIND_MAX = 7.0                           # wind speed at 10 m (m/s), random per episode
    THERMAL_W = (1.5, 4.5)                   # thermal core updraft (m/s)
    THERMAL_R = (15.0, 35.0)                 # thermal radius (m)
    THERMAL_TOP = 120.0
    SINK = 0.3                               # gentle sink between thermals (m/s)
    PHYSICS = ("massless", "wings-v1", "wings-v2")

    # ---- task parameters ----
    DT = 0.02                 # control step (50 Hz)
    SUBSTEPS = 4              # physics at 200 Hz
    MAX_STEPS = 2000          # 40 s episodes
    RING_TIMEOUT = 600        # 12 s to reach each ring
    N_OBS = 52                # wings-v2; see n_obs()
    N_ACT = 9

    @staticmethod
    def n_obs(physics="wings-v2"):
        return {"massless": 38, "wings-v1": 42, "wings-v2": 52}[physics]

    def __init__(self, num_envs, device, ring_radius=None, compile=False, species="hawk", physics="wings-v2",
                 energy_cost=0.0, energy_budget=False, soaring=False, wing_dynamics=None):
        if wing_dynamics is not None:          # older API
            physics = "wings-v1" if wing_dynamics else "massless"
        if species not in SPECIES:
            raise ValueError(f"unknown species {species!r}; choose from {', '.join(SPECIES)}")
        if physics not in self.PHYSICS:
            raise ValueError(f"unknown physics {physics!r}; choose from {', '.join(self.PHYSICS)}")
        if physics != "wings-v2" and (energy_budget or soaring):
            raise ValueError("the energy budget and soaring need physics='wings-v2'")
        for k, v in SPECIES[species].items():
            setattr(self, k, v)
        self.species = species
        self.physics = physics
        self.v2 = physics == "wings-v2"
        self.wing_dynamics = physics != "massless"
        self.energy_cost = energy_cost
        self.energy_budget = energy_budget
        self.soaring = soaring
        if self.v2:
            self.DYN = (0, 1, 2, 3, 4, 5)
        self.N_OBS = self.n_obs(physics)
        if ring_radius is None:
            ring_radius = 2 * self.RING_MIN_R
        self.n = num_envs
        self.dev = device
        self.static_masks = torch.device(device).type == "cuda"
        self._ring_radius = ring_radius
        f = dict(device=device, dtype=torch.float32)
        self.f = f
        self._ring_radius_tensor = torch.tensor(ring_radius, **f)
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
        if self.wing_dynamics:
            kin[self.dyn_idx] = False
        self.kin_mask = kin.to(torch.float32)
        if self.v2:
            self._init_v2()

        z = lambda *s: torch.zeros((num_envs,) + s, **f)
        self.pos, self.vel = z(3), z(3)
        self.quat = z(4)
        self.omega = z(3)             # body frame
        self.joints = z(9)
        self.target = z(9)
        self.qd = z(len(self.DYN))    # velocities of the joints with dynamics
        self.rdot = z(2 * self.N_SEG + 1, 3)   # surface / wing-mass velocities relative to body
        self.work = z()               # muscle work done in the last control step (J)
        self.acc = z(3)               # body-frame inertial acceleration of the body (wings-v2)
        self.dom = z(3)               # body angular acceleration (wings-v2)
        self.e_max = self.ENERGY_PER_KG * self.MASS
        self.energy = torch.full((num_envs,), self.e_max, **f)
        self.wind_ref, self.wind_dir = z(), z(2)
        self.therm = z(self.N_THERMALS, 4)     # x, y, core updraft, radius
        self.ring_pos, self.ring_n = z(3), z(3)
        self.prev_dist = z()
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.since_ring = torch.zeros_like(self.steps)
        self.rings = torch.zeros_like(self.steps)
        impl = self._substep_v2 if self.v2 else (self._substep_dyn if self.wing_dynamics else self._substep_impl)
        self._substep = self._compiled(impl, "_substep", "Physics") if compile else impl
        self._reset = self._reset_impl
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
        if compile and self.static_masks:
            self._reset = self._compiled(self._reset_impl, "_reset", "Reset")
            self._transition = self._compiled(self._transition, "_transition", "Environment transition")
            self._observe = self._compiled(self._observe, "_observe", "Observations")

    @property
    def ring_radius(self):
        return self._ring_radius

    @ring_radius.setter
    def ring_radius(self, radius):
        self._ring_radius = radius
        self._ring_radius_tensor.fill_(radius)

    def _init_v2(self):
        """Constants for the wings-v2 model."""
        f, S = self.f, self.N_SEG
        self.seg_chord = self.seg_area / (self.HALF_SPAN / S)
        sh = torch.stack([self.shoulder, self.shoulder * self.mirror])                      # [2, 3]
        self.sh2 = sh
        # neutral-pose wing mass positions -> subtract their inertia from the whole-bird value
        # each segment's mass is two equal points, ahead of and behind its centroid, so the chordwise
        # inertia (m c^2 / 12) goes through the same exact point-mass equations as everything else
        self.m_off = (self.X_MASS + 12 ** -0.5, self.X_MASS - 12 ** -0.5)
        span = torch.stack([self.ey, -self.ey])
        xs = torch.cat([o * self.seg_chord for o in self.m_off])                             # [2S]
        r0 = sh.view(2, 1, 3) + span.view(2, 1, 3) * self.seg_r.repeat(2).view(1, 2 * S, 1) + xs.view(1, 2 * S, 1) * self.ex
        m = (self.seg_mass / 2).repeat(2).view(1, 2 * S)
        Iw = torch.stack([(m * (r0[..., 1] ** 2 + r0[..., 2] ** 2)).sum(), (m * (r0[..., 0] ** 2 + r0[..., 2] ** 2)).sum(),
                          (m * (r0[..., 0] ** 2 + r0[..., 1] ** 2)).sum()])
        I_total = torch.tensor(self.INERTIA, **f)
        self.I_body = torch.maximum(I_total - Iw, 0.3 * I_total)
        self.m_wing = (self.seg_mass / 2).repeat(4)                                          # [4S]: side, point
        self.M_body = self.MASS - float(self.m_wing.sum())
        self.Mm_body = torch.diag(torch.cat([torch.full((3,), self.M_body, **f), self.I_body, torch.zeros(6, **f)])) \
            + 1e-7 * torch.eye(12, **f)
        mgL, mgc = self.MASS * G * self.HALF_SPAN, self.MASS * G * self.CHORD
        J_ft = self.J
        J_tw = (self.seg_mass * self.seg_chord ** 2 * (1 / 12 + self.X_MASS ** 2)).sum()
        wf, ws = self.DYN_FREQ
        wt = self.TWIST_FREQ
        per = lambda a, b, c: torch.tensor([a, b, c] * 2, **f)
        self.kp = per(J_ft * wf ** 2, J_ft * ws ** 2, J_tw * wt ** 2)
        self.kd = per(2 * self.DYN_DAMP * J_ft * wf, 2 * self.DYN_DAMP * J_ft * ws, 2 * self.DYN_DAMP * J_tw * wt)
        self.tau_max = per(self.FLAP_STRENGTH * mgL, 0.35 * self.FLAP_STRENGTH * mgL,
                           self.TWIST_STRENGTH * self.FLAP_STRENGTH * mgc)
        self.side_sign = torch.tensor([1.0, -1.0], **f).view(1, 2, 1)

    def _compiled(self, impl, attribute, label):
        """Compile a simulator operation with an eager fallback."""
        fast = None

        def run(*args):
            nonlocal fast
            try:
                if fast is None:
                    fast = torch.compile(impl, dynamic=False)
                    result = fast(*args)
                    print(f"{label} backend: torch.compile", flush=True)
                    return result
                return fast(*args)
            except Exception as e:  # noqa: BLE001
                msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
                print(f"{label} backend: eager (torch.compile failed: {msg})", flush=True)
                setattr(self, attribute, impl)
                return impl(*args)
        return run

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
    def _active_rows(self, mask):
        # Dynamic index shapes make CUDA wait for the host. Keep a fixed batch there.
        return slice(None) if self.static_masks else mask.nonzero(as_tuple=True)[0]

    def _write_rows(self, target, mask, rows, value):
        if self.static_masks:
            shape = (self.n,) + (1,) * (target.ndim - 1)
            target.copy_(torch.where(mask.view(shape), value, target))
        else:
            target[rows] = value

    def spawn_ring(self, mask, from_pos, heading):
        rows = self._active_rows(mask)
        origin = from_pos[rows]
        k = origin.shape[0]
        if k == 0:
            return
        yaw = heading[rows] + (torch.rand(k, **self.f) * 2 - 1) * math.radians(70)
        dist = (18 + torch.rand(k, **self.f) * 17) * max(1.0, sum(self.START_SPEED) / 20)
        dz = (torch.rand(k, **self.f) * 2 - 1) * 7
        p = origin + torch.stack([dist * torch.cos(yaw), dist * torch.sin(yaw), dz], -1)
        p[:, 2] = p[:, 2].clamp(15, 70)
        d = p - origin
        d[:, 2] *= 0.5
        self._write_rows(self.ring_pos, mask, rows, p)
        self._write_rows(self.ring_n, mask, rows, d / d.norm(dim=-1, keepdim=True))

    # ------------------------------------------------------------------ reset
    def reset(self, mask):
        self._reset(mask)

    def _reset_impl(self, mask):
        rows = self._active_rows(mask)
        k = self.pos[rows].shape[0]
        if k == 0:
            return
        f = self.f
        yaw = torch.rand(k, **f) * 2 * math.pi
        pos = torch.stack([torch.zeros(k, **f), torch.zeros(k, **f), 35 + torch.rand(k, **f) * 10], -1)
        self._write_rows(self.pos, mask, rows, pos)
        pitch = (torch.rand(k, **f) * 2 - 1) * 0.15
        roll = (torch.rand(k, **f) * 2 - 1) * 0.2
        qy = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
        qp = torch.stack([torch.cos(pitch / 2), 0 * yaw, torch.sin(pitch / 2), 0 * yaw], -1)
        qr = torch.stack([torch.cos(roll / 2), torch.sin(roll / 2), 0 * yaw, 0 * yaw], -1)
        q = quat_mul(quat_mul(qy, qp), qr)
        self._write_rows(self.quat, mask, rows, q)
        lo, hi = self.START_SPEED
        speed = lo + torch.rand(k, **f) * (hi - lo)
        fwd = quat_to_mat(q)[:, :, 0]
        self._write_rows(self.vel, mask, rows, fwd * speed.unsqueeze(-1))
        j = torch.zeros(k, 9, **f)
        j[:, 8] = 0.5
        self._write_rows(self.joints, mask, rows, j)
        self._write_rows(self.target, mask, rows, j)
        for state in (self.omega, self.qd, self.rdot, self.acc, self.dom, self.steps, self.since_ring, self.rings):
            state.masked_fill_(mask.view((self.n,) + (1,) * (state.ndim - 1)), 0)
        self.energy.masked_fill_(mask, self.e_max)
        if self.soaring:
            self._write_rows(self.wind_ref, mask, rows, torch.rand(k, **f) * self.WIND_MAX)
            wa = torch.rand(k, **f) * 2 * math.pi
            self._write_rows(self.wind_dir, mask, rows, torch.stack([torch.cos(wa), torch.sin(wa)], -1))
            self._write_rows(self.therm, mask, rows, self._new_thermals(pos, self.N_THERMALS, near=0.0))
        self.spawn_ring(mask, self.pos, torch.atan2(self.vel[:, 1], self.vel[:, 0]))
        self._write_rows(self.prev_dist, mask, rows, (self.ring_pos[rows] - pos).norm(dim=-1))

    # ------------------------------------------------------------------ soaring
    def _new_thermals(self, center, k_th, near=30.0):
        """Random thermals around `center` [n, 3] -> [n, k_th, 4]."""
        n, f = center.shape[0], self.f
        ang = torch.rand(n, k_th, **f) * 2 * math.pi
        dist = near + torch.rand(n, k_th, **f) * (110.0 - near)
        w = self.THERMAL_W[0] + torch.rand(n, k_th, **f) * (self.THERMAL_W[1] - self.THERMAL_W[0])
        rad = self.THERMAL_R[0] + torch.rand(n, k_th, **f) * (self.THERMAL_R[1] - self.THERMAL_R[0])
        return torch.stack([center[:, None, 0] + dist * torch.cos(ang), center[:, None, 1] + dist * torch.sin(ang), w, rad], -1)

    def _refresh_thermals(self):
        """Thermals far behind the bird dissipate and new ones form around it."""
        d = (self.therm[..., :2] - self.pos[:, None, :2]).norm(dim=-1)
        far = d > 130.0
        if self.static_masks or far.any():
            fresh = self._new_thermals(self.pos, self.N_THERMALS, near=40.0)
            self.therm = torch.where(far.unsqueeze(-1), fresh, self.therm)

    def wind_at(self, pos, wind_ref, wind_dir, therm):
        """World-frame wind [n, 3] at each bird's position."""
        z = pos[:, 2].clamp_min(0.3)
        prof = torch.log(z / self.Z0) / math.log(10.0 / self.Z0)
        horiz = (wind_ref * prof).unsqueeze(-1) * wind_dir
        dx = pos[:, None, 0] - therm[..., 0]
        dy = pos[:, None, 1] - therm[..., 1]
        up = (therm[..., 2] * torch.exp(-(dx * dx + dy * dy) / therm[..., 3] ** 2)).sum(-1)
        up = up * torch.sigmoid((self.THERMAL_TOP - z) / 8.0) - self.SINK
        return torch.cat([horiz, up.unsqueeze(-1)], -1)

    # ------------------------------------------------------------------ obs
    def obs(self):
        return self._observe()

    def _observe(self):
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
        if self.v2:
            parts.append((self.energy / self.e_max).unsqueeze(-1))
            if self.soaring:
                wind_b = mv(Rt, self.wind_at(self.pos, self.wind_ref, self.wind_dir, self.therm))
                rel = torch.cat([self.therm[..., :2] - self.pos[:, None, :2],
                                 torch.zeros_like(self.therm[..., :1])], -1)                     # [n, K, 3]
                near = rel[..., :2].norm(dim=-1).argmin(-1)
                idx = torch.arange(self.n, device=self.dev)
                to_th = mv(Rt, rel[idx, near])
                parts += [wind_b / 10, (to_th / 50).clamp(-3, 3), self.therm[idx, near, 2:3] / 5]
            else:
                parts.append(torch.zeros(self.n, 7, **self.f))
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

    def _wing_kin(self, joints, qd):
        """wings-v2 kinematics: surfaces, hinge axes, aero/mass points relative to the shoulders, the
        levers dpoint/dq for flap, sweep, twist, and the points' velocities relative to the body."""
        n, S = joints.shape[0], self.N_SEG
        r, c, nrm, s, area, axf, axs = self.surfaces(joints, axes=True)
        span = s[:, :2 * S].reshape(n, 2, S, 3)
        axt = -(span[:, :, 0] * self.side_sign)                                          # twist axis [n, 2, 3]
        chord = c[:, :2 * S].reshape(n, 2, S, 3)
        d = r[:, :2 * S].reshape(n, 2, S, 3) - self.sh2.view(1, 2, 1, 3)                 # on the twist axis
        cl = self.seg_chord.view(1, 1, S, 1)
        p_a = d + self.X_AERO * cl * chord                                               # aero centres
        p_m = torch.cat([d + o * cl * chord for o in self.m_off], 2)                     # mass points [n, 2, 2S, 3]
        axes = torch.stack([axf, axs, axt], 2)                                           # [n, 2, 3(joint), 3]
        lev_a = torch.cross(axes.unsqueeze(3).expand(-1, -1, -1, S, -1),
                            p_a.unsqueeze(2).expand(-1, -1, 3, -1, -1), dim=-1)        # [n, 2, 3, S, 3]
        lev_m = torch.cross(axes.unsqueeze(3).expand(-1, -1, -1, 2 * S, -1),
                            p_m.unsqueeze(2).expand(-1, -1, 3, -1, -1), dim=-1)        # [n, 2, 3, 2S, 3]
        q6 = qd.view(n, 2, 3, 1, 1)
        return (r, c, nrm, s, area, axes, p_a, p_m, lev_a, lev_m, (lev_a * q6).sum(2), (lev_m * q6).sum(2))

    def _substep_v2(self, pos, vel, quat, omega, joints, target, qd, rdot_prev, work, acc_prev, dom_prev,
                    strength, wind_ref, wind_dir, therm):
        """One physics substep of the wings-v2 model (see module docstring)."""
        h = self.DT / self.SUBSTEPS
        S = self.N_SEG
        n = pos.shape[0]
        R = quat_to_mat(quat)
        Rt = R.transpose(-1, -2)
        if self.soaring:
            v_air = mv(Rt, vel - self.wind_at(pos, wind_ref, wind_dir, therm))
        else:
            v_air = mv(Rt, vel)
        g_b = mv(Rt, self.down) * G

        # tail: light rate-limited servos
        joints = joints + (target - joints).clamp(-self.j_rate * h, self.j_rate * h) * self.kin_mask

        r, c, nrm, s, area, axes, p_a, p_m, lev_a, lev_m, rdot_a, rdot_m = self._wing_kin(joints, qd)
        # velocity-product ("bias") acceleration of the wing masses from the joints' own motion:
        # d/dt(sum_k lev_k(q) qd_k) at constant qd, by a small step along qd (exact up to O(eps))
        eps = 1e-3
        j_eps = joints.index_add(1, self.dyn_idx, qd * eps)
        rdot_m_eps = self._wing_kin(j_eps, qd)[-1]
        joint_bias = ((rdot_m_eps - rdot_m) / eps).reshape(n, 4 * S, 3)

        r_air = torch.cat([(p_a + self.sh2.view(1, 2, 1, 3)).reshape(n, 2 * S, 3), r[:, 2 * S:]], 1)
        r_dot = torch.cat([rdot_a.reshape(n, 2 * S, 3), torch.zeros_like(r[:, 2 * S:])], 1)
        F = self.aero(v_air, omega, r_air, c, nrm, s, area, r_dot, per_surface=True)     # [n, SURF, 3]

        # ---- coupled body + wing dynamics. Generalised coordinates u = [body accel (3), body angular accel (3),
        # joint accels (6)]. Every wing mass point i accelerates as a_o + dom x r_i + L_i qdd + bias_i, which
        # gives a 12x12 mass matrix solved each substep (no lagged accelerations, so it stays stable).
        P = 2 * S                                                           # aero points; mass points are 2P
        r_m = (p_m + self.sh2.view(1, 2, 1, 3)).reshape(n, 2 * P, 3)
        v_m = rdot_m.reshape(n, 2 * P, 3)

        def jac(rp, lev):                                                   # [n, points, 3, 12]
            k = lev.shape[3]
            Jw = torch.zeros(n, 2, k, 3, 6, **self.f)
            Jw[:, 0, :, :, 0:3] = lev[:, 0].permute(0, 2, 3, 1)
            Jw[:, 1, :, :, 3:6] = lev[:, 1].permute(0, 2, 3, 1)
            eye = torch.eye(3, **self.f).expand(n, 2 * k, 3, 3)
            return torch.cat([eye, -skew(rp), Jw.reshape(n, 2 * k, 3, 6)], -1)

        Jm = jac(r_m, lev_m)
        Ja = jac(r_air[:, :P], lev_a)
        m = self.m_wing.view(1, 2 * P, 1)
        om = omega.unsqueeze(1).expand_as(r_m)
        bias = torch.cross(om, torch.cross(om, r_m, dim=-1), dim=-1) + 2 * torch.cross(om, v_m, dim=-1) + joint_bias
        Mm = torch.einsum("npai,npaj->nij", Jm * m.unsqueeze(-1), Jm)
        Mm = Mm + self.Mm_body

        q = joints[:, self.dyn_idx]
        tmax = self.tau_max * strength.unsqueeze(-1)
        tau = torch.maximum(torch.minimum(self.kp * (target[:, self.dyn_idx] - q) - self.kd * qd, tmax), -tmax)
        F_tail = F[:, P:]
        drag = -0.5 * RHO * self.BODY_CDA * v_air.norm(dim=-1, keepdim=True) * v_air
        rhs = (torch.einsum("npai,npa->ni", Jm, m * (g_b.unsqueeze(1) - bias))
               + torch.einsum("npai,npa->ni", Ja, F[:, :P]))
        body = torch.cat([F_tail.sum(1) + drag + self.M_body * g_b,
                          torch.cross(r_air[:, P:], F_tail, dim=-1).sum(1)
                          - torch.cross(omega, self.I_body * omega, dim=-1)], -1)
        rhs = rhs + torch.cat([body, tau], -1)
        A = spd_inverse(Mm)                                                              # [n, 12, 12]
        u = mv(A, rhs)
        acc, dom, qdd = u[:, :3], u[:, 3:6], u[:, 6:]
        qd = (qd + qdd * h).clamp(-self.QD_MAX_V2, self.QD_MAX_V2)

        # joint stops: a plastic impact. The impulse that stops the joint is applied through the full mass
        # matrix, so the body takes up exactly the momentum the wing loses.
        lo, hi = self.j_lo[self.dyn_idx], self.j_hi[self.dyn_idx]
        qn = q + qd * h
        hit = (((qn <= lo) & (qd < 0)) | ((qn >= hi) & (qd > 0))).to(qd.dtype)          # [n, 6]
        A_jj = A[:, 6:, 6:]
        A_h = A_jj * hit.unsqueeze(-1) * hit.unsqueeze(-2) + torch.diag_embed(1 - hit)
        lam = mv(spd_inverse(A_h), -qd * hit) * hit
        dnu = mv(A[:, :, 6:], lam)                                                       # [n, 12]
        qd = (qd + dnu[:, 6:]) * (1 - hit)                                               # exact zero for hit joints
        dv_body, dw_body = dnu[:, :3], dnu[:, 3:6]
        joints = joints.index_copy(1, self.dyn_idx, torch.minimum(torch.maximum(q + qd * h, lo), hi))
        work = work + (tau * qd).abs().sum(-1) * h

        vel = vel + mv(R, acc * h + dv_body)
        pos = pos + vel * h
        omega = (omega + dom * h + dw_body).clamp(-30, 30)
        dq = quat_mul(quat, torch.cat([torch.zeros_like(omega[:, :1]), omega], -1))
        quat = quat + 0.5 * h * dq
        quat = quat / quat.norm(dim=-1, keepdim=True)
        vw = v_m
        rdot_new = rdot_prev                                                             # unused by wings-v2
        return pos, vel, quat, omega, joints, qd, rdot_new, work, acc, dom

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = action.clamp(-1, 1)
        self.target = self.j_lo + (action + 1) * 0.5 * (self.j_hi - self.j_lo)
        start_pos = self.pos.clone()
        if self.v2:
            work = torch.zeros_like(self.work)
            if self.energy_budget:
                strength = self.WEAK + (1 - self.WEAK) * (self.energy / (0.05 * self.e_max)).clamp(0, 1)
            else:
                strength = torch.ones_like(self.energy)
            for _ in range(self.SUBSTEPS):
                (self.pos, self.vel, self.quat, self.omega, self.joints, self.qd, self.rdot, work, self.acc,
                 self.dom) = self._substep(self.pos, self.vel, self.quat, self.omega, self.joints, self.target,
                                           self.qd, self.rdot, work, self.acc, self.dom, strength,
                                           self.wind_ref, self.wind_dir, self.therm)
            self.work = work
        elif self.wing_dynamics:
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

        return self._transition(start_pos, action)

    def _transition(self, start_pos, action):
        self.steps += 1
        self.since_ring += 1

        # ---- ring crossing: segment start->end crosses ring plane inside radius
        d0 = ((start_pos - self.ring_pos) * self.ring_n).sum(-1)
        d1 = ((self.pos - self.ring_pos) * self.ring_n).sum(-1)
        crossed = (d0 < 0) & (d1 >= 0)
        t = d0 / (d0 - d1 + 1e-8)
        hit_pt = start_pos + (self.pos - start_pos) * t.unsqueeze(-1)
        passed = crossed & ((hit_pt - self.ring_pos).norm(dim=-1) < self._ring_radius_tensor)

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
        self.since_ring.masked_fill_(passed, 0)
        if self.energy_budget:
            self.energy = (self.energy - self.work - self.BASAL_W_PER_KG * self.MASS * self.DT
                           + passed.float() * self.RING_FOOD * self.e_max).clamp(0, self.e_max)
        if self.soaring:
            self._refresh_thermals()
        heading = torch.atan2(self.vel[:, 1], self.vel[:, 0])
        self.spawn_ring(passed, self.ring_pos.clone(), heading)
        self.prev_dist = (self.ring_pos - self.pos).norm(dim=-1)

        terminated = crash | lost
        truncated = (self.steps >= self.MAX_STEPS) & ~terminated
        info = {"passed": passed, "rings": self.rings.clone(), "done": terminated | truncated,
                "power": self.work / self.DT if self.wing_dynamics else torch.zeros_like(reward),
                "energy": self.energy / self.e_max}
        return reward, terminated, truncated, info
