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
  * joints are rate-limited servos (the network sets targets, the joints move)

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

    # ---- task parameters ----
    DT = 0.02                 # control step (50 Hz)
    SUBSTEPS = 4              # physics at 200 Hz
    MAX_STEPS = 2000          # 40 s episodes
    RING_TIMEOUT = 600        # 12 s to reach each ring
    N_OBS = 38
    N_ACT = 9

    def __init__(self, num_envs, device, ring_radius=2.5, compile=False):
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

        z = lambda *s: torch.zeros((num_envs,) + s, **f)
        self.pos, self.vel = z(3), z(3)
        self.quat = z(4)
        self.omega = z(3)             # body frame
        self.joints = z(9)
        self.target = z(9)
        self.ring_pos, self.ring_n = z(3), z(3)
        self.prev_dist = z()
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.since_ring = torch.zeros_like(self.steps)
        self.rings = torch.zeros_like(self.steps)
        self._substep = self._substep_impl
        if compile:
            self._substep = torch.compile(self._substep_impl, dynamic=False)
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    # ------------------------------------------------------------------ geometry
    def surfaces(self, joints):
        """Return per-surface (position r, chord dir c, normal n, span dir s, area)
        in body frame. Shape [N, S, 3] with S = 2*N_SEG + 1."""
        n = joints.shape[0]
        out_r, out_c, out_n, out_s, out_a = [], [], [], [], []
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
        tp, tt, ts = joints[:, 6], joints[:, 7], joints[:, 8]
        Rt = rot_x(tt) @ rot_y(-tp)
        c, s, nn = Rt[..., :, 0], Rt[..., :, 1], Rt[..., :, 2]
        out_r.append(self.tail_pos.expand(n, 1, 3))
        out_c.append(c.unsqueeze(1)); out_n.append(nn.unsqueeze(1)); out_s.append(s.unsqueeze(1))
        out_a.append((self.TAIL_AREA[0] + (self.TAIL_AREA[1] - self.TAIL_AREA[0]) * ts).unsqueeze(1))
        return (torch.cat(out_r, 1), torch.cat(out_c, 1), torch.cat(out_n, 1),
                torch.cat(out_s, 1), torch.cat(out_a, 1))

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

    def aero(self, vel_b, omega, r, c, nrm, s, area, r_dot):
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
        return F.sum(1), torque

    # ------------------------------------------------------------------ rings
    def spawn_ring(self, mask, from_pos, heading):
        k = int(mask.sum())
        if k == 0:
            return
        yaw = heading[mask] + (torch.rand(k, **self.f) * 2 - 1) * math.radians(70)
        dist = 18 + torch.rand(k, **self.f) * 17
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
        speed = 8 + torch.rand(k, **f) * 3
        fwd = quat_to_mat(q)[:, :, 0]
        self.vel[mask] = fwd * speed.unsqueeze(-1)
        self.omega[mask] = 0
        j = torch.zeros(k, 9, **f)
        j[:, 8] = 0.5
        self.joints[mask] = j
        self.target[mask] = j
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
        return torch.cat([
            v_b / 10, self.omega / 5, g_b,
            to_ring / dist.clamp_min(1e-3), dist / 20, to_ring.clamp(-30, 30) / 20,
            ring_n_b, self.pos[:, 2:3] / 40,
            jn, tn,
        ], -1)

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

    # ------------------------------------------------------------------ step
    def step(self, action):
        action = action.clamp(-1, 1)
        self.target = self.j_lo + (action + 1) * 0.5 * (self.j_hi - self.j_lo)
        start_pos = self.pos.clone()
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

        crash = (self.pos[:, 2] < 0) | (self.pos[:, 2] > 150) | ~torch.isfinite(self.pos).all(-1)
        lost = self.since_ring > self.RING_TIMEOUT
        reward = reward - 5.0 * crash.float()

        self.rings += passed.long()
        self.since_ring[passed] = 0
        heading = torch.atan2(self.vel[:, 1], self.vel[:, 0])
        self.spawn_ring(passed, self.ring_pos.clone(), heading)
        self.prev_dist = (self.ring_pos - self.pos).norm(dim=-1)

        terminated = crash | lost
        truncated = (self.steps >= self.MAX_STEPS) & ~terminated
        info = {"passed": passed, "rings": self.rings.clone(), "done": terminated | truncated}
        return reward, terminated, truncated, info
