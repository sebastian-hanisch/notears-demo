"""Gegenproben und Vergleichsmaße: Varsortability, Rangbaseline, der Verlust zweier Richtungen in geschlossener Form (gegen eine Regression), Standardisierung, compare."""

import numpy as np
import pytest

import nts_baselines as B
import stl_world as W
from kausal_generator import SplitMix64


def test_standardize_gives_mean_zero_and_unit_std():
    x = W.simulate(W.World(), 500, SplitMix64(1)) * np.array([1, 5, 0.2, 3, 9]) + 7
    z = B.standardize(x)
    assert z.mean(axis=0) == pytest.approx(0.0, abs=1e-12) and z.std(axis=0) == pytest.approx(1.0)


def test_reach_matrix_follows_paths():
    r = B.reach_matrix({(0, 1), (1, 2)}, 4)
    assert r[0, 1] and r[1, 2] and r[0, 2] and not r[2, 0] and not r[0, 3] and not r[3, 3] and not r[0, 0]


def test_varsortability_by_hand():
    x_up = np.column_stack([np.array([-1.0, 1.0, -1.0, 1.0]), np.array([-2.0, 2.0, -2.0, 2.0])])        # Var 1 und 4
    assert B.varsortability({(0, 1)}, x_up) == 1.0 and B.varsortability({(1, 0)}, x_up) == 0.0
    assert B.varsortability({(0, 1)}, B.standardize(x_up)) == 0.5
    x3 = np.column_stack([np.array([-1.0, 1.0] * 2), np.array([-3.0, 3.0] * 2), np.array([-2.0, 2.0] * 2)])    # Var 1, 9, 4: Kette 0 → 1 → 2
    assert B.varsortability({(0, 1), (1, 2)}, x3) == pytest.approx((1 + 0 + 1) / 3)                          # Paare (0,1) ja, (1,2) nein, (0,2) ja
    assert np.isnan(B.varsortability(set(), x3))


def test_varsortability_of_the_world():
    world = W.World()
    truth = {(p, c) for c, ps in enumerate(world.parents) for p in ps}
    x = W.simulate(world, 2000, SplitMix64(2))
    assert B.varsortability(truth, x) == 1.0 and B.varsortability(truth, B.standardize(x)) == 0.5


def test_pair_losses_by_hand(chain_cov):
    r = B.pair_losses(chain_cov, 0, 1)
    assert r["loss_forward"] == pytest.approx(1.0) and r["loss_backward"] == pytest.approx(0.5 * (1.64 + 1 - 0.64 / 1.64)) and r["loss_backward"] == pytest.approx(1.1249, abs=5e-5)
    assert r["difference"] == pytest.approx(-0.1249, abs=5e-5) and r["rho2"] == pytest.approx(0.64 / 1.64) and r["difference"] == pytest.approx(0.5 * r["rho2"] * (1 - 1.64))
    swapped = B.pair_losses(chain_cov, 1, 0)
    assert swapped["loss_forward"] == pytest.approx(r["loss_backward"]) and swapped["difference"] == pytest.approx(-r["difference"])


def test_pair_losses_tie_when_standardized():
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    r = B.pair_losses(cov, 0, 1)
    assert r["difference"] == pytest.approx(0.0, abs=1e-15) and r["loss_forward"] == pytest.approx(0.5 * (2 - 0.36))


def test_closed_form_difference_holds_for_random_covariances():
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.normal(size=(2, 2))
        cov = a @ a.T + 0.1 * np.eye(2)
        r = B.pair_losses(cov, 0, 1)
        assert r["difference"] == pytest.approx(0.5 * r["rho2"] * (cov[0, 0] - cov[1, 1]), abs=1e-12)


def test_pair_losses_equal_the_losses_of_two_regressions():
    """Zweiter Weg: Verlust einer Richtung = ½ (Var(Ursache) + Restvarianz der Wirkung nach Regression auf die Ursache), aus den Daten."""
    x = W.simulate(W.World(), 5000, SplitMix64(3))[:, [W.Z, W.X]]
    x = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False, bias=True)
    r = B.pair_losses(cov, 0, 1)
    for cause, effect, key in ((0, 1, "loss_forward"), (1, 0, "loss_backward")):
        beta = (x[:, cause] @ x[:, effect]) / (x[:, cause] @ x[:, cause])
        resid = x[:, effect] - beta * x[:, cause]
        assert r[key] == pytest.approx(0.5 * (x[:, cause].var() + resid.var()), abs=1e-12)


def test_sortnregress_orders_by_variance_and_finds_the_chain():
    b = np.zeros((3, 3))
    b[1, 0], b[2, 1] = 0.8, 0.8
    x = W.simulate_linear(b, 4000, SplitMix64(7))
    assert B.sortnregress(x) == {(0, 1), (1, 2)}
    flipped = x[:, ::-1]                                                          # Spalten umgekehrt: dieselben Daten, andere Nummern
    assert B.sortnregress(flipped) == {(2, 1), (1, 0)}


def test_sortnregress_edges_depend_only_on_the_variance_order():
    x = W.simulate(W.World(), 3000, SplitMix64(9))
    assert B.sortnregress(x) == {(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)}
    rescaled = x * np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    assert B.sortnregress(rescaled) == B.sortnregress(x)
    swapped = x.copy()
    swapped[:, W.S] *= 3.0                                                          # Var(S) = 9 über Var(Z): S kommt nach Z
    assert (W.Z, W.S) in B.sortnregress(swapped) and (W.S, W.Z) not in B.sortnregress(swapped)


def test_sortnregress_threshold_removes_weak_edges():
    x = np.column_stack([SplitMix64(1).normal(2000), SplitMix64(2).normal(2000) * 1.5])
    assert B.sortnregress(x) == set() and B.sortnregress(x, t_min=0.0) != set()


def test_compare_by_hand():
    truth = {(0, 1), (1, 2), (2, 3)}
    c = B.compare({(0, 1), (2, 1), (0, 3)}, {(2, 3)}, truth)
    assert (c["correct"], c["reversed"], c["undirected"], c["missing"], c["extra"], c["n_true"]) == (1, 1, 1, 0, 1, 3) and c["shd"] == 1 + 1 + 0.5 + 0
    perfect = B.compare(set(truth), set(), truth)
    assert perfect["shd"] == 0 and perfect["correct"] == 3
    empty = B.compare(set(), set(), truth)
    assert empty["missing"] == 3 and empty["shd"] == 3


def test_variance_order_by_hand_and_with_ties():
    x = np.column_stack([np.array([-3.0, 3.0] * 2), np.array([-1.0, 1.0] * 2), np.array([-2.0, 2.0] * 2)])          # Var 9, 1, 4
    assert B.variance_order(x) == [1, 2, 0]
    z = B.standardize(x)
    orders = {tuple(B.variance_order(z, seed)) for seed in range(30)}
    assert all(sorted(o) == [0, 1, 2] for o in orders) and len(orders) == 6                                         # bei Gleichstand: feste Zufallsreihenfolge, alle sechs kommen vor
    assert B.variance_order(z, 5) == B.variance_order(z, 5) and B.variance_order(z, 5) == B.variance_order(z * (1 + 1e-14), 5)    # unempfindlich gegen Rundungsrauschen


def test_ties_do_not_follow_the_column_number():
    """Auf standardisierten Daten darf die Rangbaseline die Reihenfolge der Spalten nicht verraten: Über viele Startwerte kommt jede Ordnung der Welt vor."""
    z = B.standardize(W.simulate(W.World(), 500, SplitMix64(2)))
    firsts = {B.variance_order(z, seed)[0] for seed in range(40)}
    assert firsts == {0, 1, 2, 3, 4}
