import unittest

import torch

from bird_env import BirdEnv


class MaskedEnvironmentTest(unittest.TestCase):
    def test_ring_radius_change_applies_to_next_transition(self):
        env = BirdEnv(1, "cpu", physics="massless")
        env.pos[:] = torch.tensor([1.0, 1.5, 40.0])
        env.ring_pos[:] = torch.tensor([0.0, 0.0, 40.0])
        env.ring_n[:] = torch.tensor([1.0, 0.0, 0.0])
        start = torch.tensor([[-1.0, 1.5, 40.0]])
        action = torch.zeros(1, BirdEnv.N_ACT)

        env.ring_radius = 1.0
        self.assertFalse(env._transition(start, action)[3]["passed"].item())
        env.ring_radius = 2.0
        self.assertTrue(env._transition(start, action)[3]["passed"].item())

    def test_fixed_shape_reset_preserves_active_birds(self):
        env = BirdEnv(3, "cpu", energy_budget=True, soaring=True)
        env.static_masks = True  # exercise the CUDA update path without requiring a GPU
        env.steps.fill_(37)
        env.since_ring.fill_(12)
        env.rings.fill_(2)
        env.energy.fill_(11)
        fields = ("pos", "vel", "quat", "joints", "ring_pos", "ring_n", "prev_dist", "steps",
                  "since_ring", "rings", "energy", "wind_ref", "wind_dir", "therm")
        before = {name: getattr(env, name).clone() for name in fields}
        done = torch.tensor([False, True, False])
        env.reset(done)

        for name, value in before.items():
            torch.testing.assert_close(getattr(env, name)[~done], value[~done])
        self.assertEqual(env.steps[1].item(), 0)
        self.assertEqual(env.since_ring[1].item(), 0)
        self.assertEqual(env.rings[1].item(), 0)
        self.assertEqual(env.energy[1].item(), env.e_max)
        self.assertGreaterEqual(env.pos[1, 2].item(), 35)
        torch.testing.assert_close(env.ring_n[1].norm(), torch.tensor(1.0))
        torch.testing.assert_close(env.prev_dist[1], (env.ring_pos[1] - env.pos[1]).norm())

    def test_fixed_shape_ring_spawn_preserves_unselected_rings(self):
        env = BirdEnv(3, "cpu")
        env.static_masks = True
        old_pos, old_normal = env.ring_pos.clone(), env.ring_n.clone()
        passed = torch.tensor([True, False, True])
        env.spawn_ring(passed, env.ring_pos, torch.zeros(3))
        torch.testing.assert_close(env.ring_pos[~passed], old_pos[~passed])
        torch.testing.assert_close(env.ring_n[~passed], old_normal[~passed])
        self.assertTrue(((env.ring_pos - old_pos)[passed, 0] > 0).all())
        torch.testing.assert_close(env.ring_n[passed].norm(dim=-1), torch.ones(2))
