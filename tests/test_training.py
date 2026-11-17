import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class TrainingSmokeTest(unittest.TestCase):
    def train(self, out, *args):
        cmd = [sys.executable, "train.py", "--out", str(out), "--device", "cpu", "--no-compile",
               "--physics", "massless", "--steps", "1", "--iters", "1", "--epochs", "1", *args]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # A one-step rollout has no episode metrics, but must still save the policy.
        self.assertNotIn("rings/ep", result.stdout)
        return torch.load(out / "latest.pt", map_location="cpu", weights_only=True)

    def assert_optimizer_steps(self, ck, expected):
        steps = {state["step"].item() for state in ck["opt"]["state"].values()}
        self.assertEqual(steps, {expected})
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in ck["model"].values()))

    def test_single_sample_updates_and_resume_advances_iteration(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            ck = self.train(out, "--envs", "1")
            self.assertEqual(ck["iter"], 0)
            self.assert_optimizer_steps(ck, 1)
            saved_lr = ck["opt"]["param_groups"][0]["lr"]

            ck = self.train(out, "--envs", "1", "--resume", str(out / "latest.pt"))
            self.assertEqual(ck["iter"], 1)
            self.assert_optimizer_steps(ck, 2)
            self.assertAlmostEqual(ck["opt"]["param_groups"][0]["lr"], 1.5 * saved_lr)

    def test_final_short_minibatch_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            ck = self.train(Path(temp), "--envs", "3", "--mb-size", "2")
            self.assert_optimizer_steps(ck, 2)


if __name__ == "__main__":
    unittest.main()
