"""Meaningful regression checks for time-scale and nonlinear solver corrections."""
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from tsvolterra.timescale import TimeScale,Partition
from tsvolterra import nystrom
from tsvolterra.quadrature import trapezoid_weights,kappa_step
from revision_checks import irregular_problem,exact_defect
from revision_neural import analytic_certificate,network


class RevisionTests(unittest.TestCase):
    def test_scattered_panel_exact(self):
        T=TimeScale.from_points([1,3,9])
        self.assertEqual(trapezoid_weights(T,3,9),(6,0))
        self.assertEqual(kappa_step(T,3,9),0)

    def test_invalid_partition(self):
        T=TimeScale([(0,1),(2,3)])
        for pts in [[0,.5,.4,3],[0,1.5,3],[.5,1,3]]:
            with self.assertRaises(ValueError):Partition(T,np.array(pts))

    def test_coarsening_requires_discrete_scale(self):
        with self.assertRaises(ValueError):Partition.coarse(TimeScale([(0,1)]),2)

    def test_exact_cubic_defect_and_error_bound(self):
        prob,p,x,integ=irregular_problem()
        P=Partition.gap_conforming(prob.T,h=.05)
        z=nystrom.residual(prob,P,x(P.t))
        self.assertLess(np.max(abs(z-exact_defect(P))),2e-15)
        U=nystrom.majorant_stability(P,lambda t,s:.18*(1+.07*t*s))
        Y=nystrom.solve(prob,P)
        self.assertLess(np.max(abs(Y-x(P.t))),U[-1]*np.max(abs(exact_defect(P)))+2e-15)

    def test_newton_failure_is_reported(self):
        prob,*_=irregular_problem()
        P=Partition.gap_conforming(prob.T,h=.05)
        with self.assertRaises(RuntimeError):nystrom.solve(prob,P,maxit=0,fp_maxit=0)

    def test_analytic_certificate_is_exact_rational(self):
        d=analytic_certificate()
        self.assertTrue(d['validated'])
        self.assertEqual(d['exact_bound_numerator'],243520131)
        self.assertEqual(d['exact_bound_denominator'],200000000000)

    def test_training_parameter_jacobian(self):
        rng=np.random.default_rng(6);m=3;theta=rng.normal(size=1+3*m);t=np.array([.2,.7,2.])
        y,J=network(t,theta,m,3.55)
        for k in range(len(theta)):
            d=np.zeros_like(theta);d[k]=1e-6
            yp,_=network(t,theta+d,m,3.55);ym,_=network(t,theta-d,m,3.55)
            self.assertLess(np.max(abs((yp-ym)/2e-6-J[:,k])),2e-9)


if __name__=='__main__':unittest.main()
