"""NOTEARS von Grund auf: Matrixexponential, h(W) und Gradient von Hand und gegen zweite Wege, Schwelle, Löser auf einfachen Instanzen, Verlauf des augmentierten Lagrange-Verfahrens."""

import math

import numpy as np
import pytest
import scipy.linalg as sla

import nts_notears as N
import stl_world as W
from kausal_generator import SplitMix64


def test_expm_matches_scipy_for_small_and_large_norms():
    rng = np.random.default_rng(0)
    for scale in (0.01, 0.5, 3.0, 12.0):
        a = rng.normal(size=(6, 6)) * scale
        assert N.expm(a) == pytest.approx(sla.expm(a), rel=1e-9, abs=1e-9)
    assert N.expm(np.zeros((3, 3))) == pytest.approx(np.eye(3))
    assert N.expm(np.array([[0.0, 1.0], [0.0, 0.0]])) == pytest.approx(np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_h_by_hand():
    assert N.h_value(np.array([[0, 1.0], [1.0, 0]])) == pytest.approx(2 * math.cosh(1) - 2) and N.h_value(np.array([[0, 1.0], [1.0, 0]])) == pytest.approx(1.0862, abs=5e-5)
    cycle3 = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    assert N.h_value(cycle3) == pytest.approx(math.e + 2 * math.exp(-0.5) * math.cos(math.sqrt(3) / 2) - 3) and N.h_value(cycle3) == pytest.approx(0.5042, abs=5e-5)
    assert N.h_value(np.array([[0, 1.0], [0, 0]])) == pytest.approx(0.0, abs=1e-12) and N.h_value(np.zeros((4, 4))) == 0.0
    for a_, b_ in ((0.5, 0.8), (1.2, 0.3)):
        assert N.h_value(np.array([[0, a_], [b_, 0]])) == pytest.approx(2 * math.cosh(a_ * b_) - 2)


def test_h_is_zero_for_dags_and_positive_with_cycles():
    rng = np.random.default_rng(1)
    for _ in range(20):
        b = np.triu(rng.normal(size=(6, 6)), 1)                   # strikt oberes Dreieck: ein DAG
        perm = rng.permutation(6)
        assert N.h_value(b[np.ix_(perm, perm)]) == pytest.approx(0.0, abs=1e-9)
    w = np.triu(rng.normal(size=(5, 5)), 1)
    w[4, 0] = 0.7                                               # schließt Kreise
    assert N.h_value(w) > 1e-3


def test_gradient_equals_finite_differences():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(4, 4)) * 0.5
    h0, g = N.h_and_grad(w)
    assert h0 == pytest.approx(N.h_value(w))
    num = np.zeros_like(w)
    for i in range(4):
        for j in range(4):
            e = np.zeros_like(w)
            e[i, j] = 1e-6
            num[i, j] = (N.h_value(w + e) - N.h_value(w - e)) / 2e-6
    assert g == pytest.approx(num, abs=1e-6)
    assert np.all(N.h_and_grad(np.zeros((3, 3)))[1] == 0.0)


def test_soft_threshold_and_acyclicity():
    assert N.soft_threshold(np.array([-2.0, -0.3, 0.0, 0.3, 2.0]), 0.5) == pytest.approx([-1.5, 0.0, 0.0, 0.0, 1.5])
    assert N.is_acyclic({(0, 1), (1, 2), (0, 2)}) and not N.is_acyclic({(0, 1), (1, 2), (2, 0)}) and not N.is_acyclic({(0, 1), (1, 0)}) and N.is_acyclic(set())


def test_threshold_removes_the_weakest_edge_of_a_cycle():
    w = np.zeros((3, 3))
    w[0, 1], w[1, 0] = 0.9, 0.5
    assert N.threshold_dag(w, 0.3) == {(0, 1)}
    w3 = np.zeros((3, 3))
    w3[0, 1], w3[1, 2], w3[2, 0] = 0.9, 0.8, 0.4
    assert N.threshold_dag(w3, 0.3) == {(0, 1), (1, 2)}
    assert N.threshold_dag(w3, 0.5) == {(0, 1), (1, 2)} and N.threshold_dag(w3, 0.85) == {(0, 1)} and N.threshold_dag(np.zeros((3, 3)), 0.3) == set()


def test_solver_recovers_a_chain_and_stays_acyclic():
    b = np.zeros((3, 3))
    b[1, 0], b[2, 1] = 0.8, 0.8
    x = W.simulate_linear(b, 2000, SplitMix64(4))
    res = N.notears(x)
    assert N.threshold_dag(res.w, 0.3) == {(0, 1), (1, 2)} and res.h < 1e-3
    assert res.w[0, 1] == pytest.approx(0.8, abs=0.15) and res.w[1, 2] == pytest.approx(0.8, abs=0.15) and np.all(np.diag(res.w) == 0)


def test_solver_returns_nothing_for_independent_columns():
    x = SplitMix64(5).normal(3000 * 4).reshape(3000, 4)
    assert N.threshold_dag(N.notears(x).w, 0.3) == set()


def test_trace_of_the_augmented_lagrangian():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    res = N.notears(x)
    rho = [t["rho"] for t in res.trace]
    assert rho[0] == 1.0 and res.trace[0]["alpha"] == 0.0 and rho == sorted(rho) and rho[-1] > 1e3 and len(res.trace) >= 5
    assert res.trace[0]["h"] > 0.05 and res.trace[-1]["h"] < 1e-3 and res.trace[-1]["h"] == pytest.approx(res.h)
    assert res.trace[-1]["loss"] > res.trace[0]["loss"] and all(t["alpha"] >= 0 for t in res.trace) and res.steps > 100


def test_without_the_constraint_the_graph_has_cycles():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    res = N.notears(x, constrained=False)
    edges = {(i, j) for i in range(5) for j in range(5) if abs(res.w[i, j]) > 0.3}
    assert not N.is_acyclic(edges) and res.h > 0.05 and len(res.trace) == 1 and res.trace[0]["rho"] == 0.0


def test_data_are_centred_inside_the_solver():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    a = N.notears(x).w
    b = N.notears(x + np.array([5.0, -3.0, 2.0, 7.0, 1.0])).w
    assert a == pytest.approx(b, abs=1e-8)


def test_stronger_penalty_shrinks_the_weights():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    small, large = N.notears(x, lam=0.01).w, N.notears(x, lam=0.3).w
    assert np.abs(large).sum() < np.abs(small).sum() and large[W.W, W.Y] == pytest.approx(0.28, abs=0.03) and small[W.W, W.Y] == pytest.approx(0.6, abs=0.1)


def test_rho_is_never_solved_at_the_limit():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    res = N.notears(x, rho_max=100.0)
    rho = [t["rho"] for t in res.trace]
    assert max(rho) == 10.0 and all(r < 100.0 for r in rho) and rho == sorted(rho)


def test_step_budget_stops_the_run():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    res = N.notears(x, max_steps=50)
    assert res.steps == 50 and len(res.trace) == 1
    assert N.notears(x).steps < 5000


def test_inner_step_limit_is_per_subproblem():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    res = N.notears(x, max_inner=1)
    assert res.steps == len(res.trace) and len(res.trace) >= 2


def test_a_loose_h_tolerance_stops_after_the_first_round():
    x = W.simulate(W.World(), 1000, SplitMix64(35))
    res = N.notears(x, h_tol=10.0)
    assert len(res.trace) == 1 and res.trace[0]["rho"] == 1.0 and len(N.notears(x).trace) > 5


def test_weight_exactly_at_the_threshold_is_not_an_edge():
    w = np.zeros((3, 3))
    w[0, 1], w[1, 2] = 0.3, 0.30001
    assert N.threshold_dag(w, 0.3) == {(1, 2)}
