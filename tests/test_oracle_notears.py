"""Dauerhaftes Orakel: Der eigene Löser (proximaler Gradient, numpy) gegen die Fassung der Originalarbeit (W = W⁺ − W⁻, L-BFGS-B aus scipy, scipy.linalg.expm) auf denselben Daten;
dazu die Schwelle und das Ergebnis auf der Welt. Beide lösen dasselbe nichtkonvexe Problem lokal; verglichen werden die Kantenmengen nach der Schwelle."""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.optimize as sopt

import nts_notears as N
import stl_world as W
from kausal_generator import SplitMix64


def reference_notears(x: np.ndarray, lam: float = 0.1, max_iter: int = 100, h_tol: float = 1e-8, rho_max: float = 1e16) -> np.ndarray:
    """Die Fassung der Originalarbeit (Zheng u. a. 2018) mit L-BFGS-B, zur Gegenprobe."""
    x = x - x.mean(axis=0)
    n, d = x.shape

    def adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def func(w):
        wm = adj(w)
        loss = 0.5 / n * ((x - x @ wm) ** 2).sum()
        e = sla.expm(wm * wm)
        h = np.trace(e) - d
        obj = loss + 0.5 * rho * h * h + alpha * h + lam * w.sum()
        g_obj = -1.0 / n * x.T @ (x - x @ wm) + (rho * h + alpha) * e.T * wm * 2
        return obj, np.concatenate((g_obj + lam, -g_obj + lam), axis=None)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            w_new = sol.x
            h_new = np.trace(sla.expm(adj(w_new) * adj(w_new))) - d
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    return adj(w_est)


def test_the_world_gives_the_same_graph_as_the_reference():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    mine, ref = N.notears(x).w, reference_notears(x)
    assert N.threshold_dag(mine, 0.3) == N.threshold_dag(ref, 0.3) == {(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)}
    assert mine == pytest.approx(ref, abs=0.03)


def test_random_dags_give_the_same_edges_as_the_reference():
    rng = SplitMix64(5)
    same = 0
    total = 10
    for r in range(total):
        b = W.random_dag(6, 0.4, rng)
        x = W.simulate_linear(b, 1000, SplitMix64(100 + r))
        same += N.threshold_dag(N.notears(x).w, 0.3) == N.threshold_dag(reference_notears(x), 0.3)
    assert same >= 9


def test_standardized_world_gives_the_same_graph_as_the_reference():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    mine, ref = N.threshold_dag(N.notears(z).w, 0.3), N.threshold_dag(reference_notears(z), 0.3)
    assert mine == ref and len(mine) >= 3
