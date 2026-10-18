"""tsvolterra: time-scale Nystrom methods and certified neural solvers for
nonlinear Volterra dynamic integral equations on time scales.

Companion code for: M. Khuddush and S. G. Georgiev, "Certified neural
approximation of nonlinear Volterra integral equations on time scales".
"""
from .timescale import TimeScale, Partition
from . import quadrature, nystrom, certify

__all__ = ["TimeScale", "Partition", "quadrature", "nystrom", "certify"]
