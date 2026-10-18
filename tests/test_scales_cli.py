"""Tests for the unusual time scales, the truncation bound and the command-line interface."""
import os, sys, subprocess, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tsvolterra import scales, nystrom
from tsvolterra.timescale import Partition
from tsvolterra.problems import VolterraProblem, kernel_exp_sin, x_star


class TestScales(unittest.TestCase):
    def test_cantor_structure(self):
        T = scales.cantor(4)
        g = sorted(round(p2 - q1, 12) for q1, p2 in T.gaps())
        self.assertEqual(T.m, 16)
        self.assertEqual(len(g), 15)
        self.assertAlmostEqual(g[-1], 1.0); self.assertAlmostEqual(g[0], 1 / 27)

    def test_sigma_mu_harmonic(self):
        T = scales.harmonic(50)
        t = T.p[9]                                # H_10
        self.assertAlmostEqual(T.mu(t), 1 / 11, places=12)
        self.assertEqual(T.sigma(T.b), T.b)       # sigma(max T) = max T

    def test_full_partition_exact_on_discrete_scale(self):
        T = scales.sqrt_scale(60); K, Kx, L = kernel_exp_sin(1, 1)
        f = lambda t: 1 + 0.5 * np.sin(2 * np.asarray(t))
        X = nystrom.solve(VolterraProblem(T, K, Kx, L, f=f), Partition(T, T.p.copy()))
        t = T.p; mu = np.diff(t); x = np.empty(len(t)); x[0] = f(t[0])
        for n in range(1, len(t)):
            x[n] = f(t[n]) + np.sum(mu[:n] * np.exp(-(t[n] - t[:n])) * np.sin(x[:n]))
        self.assertLess(np.max(abs(X - x)), 1e-13)

    def test_second_order_on_cantor(self):
        T = scales.cantor(3); K, Kx, L = kernel_exp_sin(1, 1)
        prob = VolterraProblem(T, K, Kx, L, x_exact=x_star, n_gauss=20, panel=0.02); e = []
        for k in (1, 2):
            P = Partition.gap_conforming(T, counts=[2 ** k] * T.m)
            e.append(np.max(abs(nystrom.solve(prob, P) - x_star(P.t))))
        self.assertAlmostEqual(np.log2(e[0] / e[1]), 2.0, delta=0.05)

    def test_truncation_bound(self):
        K, _, L = kernel_exp_sin(1, 1); f = lambda s: 1 + 0.5 * np.sin(2 * s)
        tr, Xr, _ = scales.qacc_solve(f, K, n=400)
        for n in (20, 60):
            t, X, mu = scales.qacc_solve(f, K, n=n)
            err = np.max(abs(X - Xr[-(n + 1):])); bound = 0.9 ** n * np.prod(1 + L * mu[:-1])
            self.assertLess(err, bound)


class TestCLI(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "tsvolterra.cli", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    def test_solve_hybrid(self):
        out = self.run_cli("solve", "--scale", "hybrid", "--h", "0.0625", "--manufactured")
        self.assertIn("47 nodes", out); self.assertIn("3.223e-04", out)     # Table 1, hybrid, h = 2^-4

    def test_info_and_scales(self):
        self.assertIn("right-scattered", self.run_cli("info", "--scale", "irr"))
        self.assertIn("cantor", self.run_cli("scales"))


if __name__ == "__main__":
    unittest.main()


class TestNewtonFallback(unittest.TestCase):
    def test_fixed_point_fallback(self):
        """with maxit=0 Newton is skipped, so the fixed-point fallback must reproduce the Newton solution"""
        import numpy as np
        from tsvolterra import nystrom
        from tsvolterra.timescale import TimeScale, Partition
        from tsvolterra.problems import VolterraProblem, kernel_exp_sin
        T = TimeScale([(0.0, 1.0), (1.25, 1.25), (2.0, 3.0)])
        K, Kx, L = kernel_exp_sin(1.0, 1.0)
        prob = VolterraProblem(T, K, Kx, L, f=lambda t: 1.0 + 0.5 * np.sin(2.0 * np.asarray(t)))
        P = Partition.gap_conforming(T, counts=[16, 0, 16])
        X1 = nystrom.solve(prob, P); X2 = nystrom.solve(prob, P, maxit=0)
        self.assertLess(float(np.max(np.abs(X1 - X2))), 1e-13)


class TestCheckpoints(unittest.TestCase):
    def test_p5_checkpoints_reload(self):
        """every stored P5 checkpoint rebuilds its trial function; random features are regenerated exactly"""
        import glob, json
        from tsvolterra import checkpoints
        from tsvolterra.problems import ts_hybrid
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = sorted(glob.glob(os.path.join(root, "results", "checkpoints", "*.npz")))
        self.assertEqual(len(files), 58)
        T = ts_hybrid(); xa = 1.0 + 0.5 * np.sin(2.0 * T.a)
        for f in files:
            tr, p, meta = checkpoints.load(f, T, xa)
            v = tr.numpy_fn(p)(np.array([0.0, 0.5, 1.25, 4.5]))
            self.assertTrue(np.all(np.isfinite(v)))
            if meta["spec"]["kind"] != "pwcheb":                     # pwcheb enforces x(a) only through the residual
                self.assertAlmostEqual(float(v[0]), xa, places=12)
            self.assertEqual(int(meta["n_params"]), int(tr.n_params))
