"""Auswertung: Simulation mit Rauschstärke und Skala, Kantenurteile, Analyse, die vier Experimente in kleinem Maßstab (Struktur und Identitäten), Baukasten für h(W)."""

import numpy as np
import pytest

import nts_constants as C
import nts_evaluation as ev
import nts_notears as N
import stl_world as W
from kausal_generator import SplitMix64


def test_number_formats():
    assert ev.de(1.5) == "1,50" and ev.de(-0.25, 3) == "−0,250" and ev.signed(0.5) == "+0,50" and ev.signed(-0.5) == "−0,50" and ev.signed(0.0) == "+0,00" and ev.pct(0.123, 1) == "12,3 %" and ev._n(10000) == "10 000"


def test_simulation_variances_and_scale():
    world = W.World()
    data = ev.simulate_scaled(world, 200000, SplitMix64(1))
    v = data.var(axis=0)
    assert v == pytest.approx([1.0, 1.64, 2.0496, 3.6329, 1.0], rel=0.02)
    wide = ev.simulate_scaled(world, 200000, SplitMix64(1), sigma=2.0).var(axis=0)
    assert wide[W.S] == pytest.approx(4.0, rel=0.02) and wide[W.Z] == pytest.approx(0.64 * 4 + 1, rel=0.02) and wide[W.W] == pytest.approx(1.0, rel=0.02)
    std = ev.simulate_scaled(world, 1000, SplitMix64(1), 1.0, True)
    assert std.mean(axis=0) == pytest.approx(0.0, abs=1e-12) and std.std(axis=0) == pytest.approx(1.0)
    assert np.array_equal(ev.simulate_scaled(world, 50, SplitMix64(3)), W.simulate(world, 50, SplitMix64(3)))             # sigma = 1 ist die Welt aus Stück 13


def test_truth_edges_and_status():
    truth = ev.truth_edges(W.World())
    assert truth == {(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)}
    got = {(a, b): st for a, b, st in ev.edge_status({(0, 1), (2, 1), (4, 0)}, truth)}
    assert got[(0, 1)] == "richtig" and got[(2, 1)] == "umgekehrt" and got[(4, 0)] == "überzählig" and got[(1, 3)] == "fehlt" and got[(2, 3)] == "fehlt" and got[(4, 3)] == "fehlt"
    assert all(st == "richtig" for _, _, st in ev.edge_status(truth, truth))


def test_settings_property():
    assert not ev.Settings().standardized and ev.Settings(scale="standardisiert").standardized


def test_analysis_of_the_default_setting():
    a = ev.analyse(ev.Settings())
    assert a.learned == a.truth == {(0, 1), (1, 2), (1, 3), (2, 3), (4, 3)} and a.rank == a.truth and all(st == "richtig" for _, _, st in a.status)
    assert a.compare["NOTEARS"]["shd"] == 0 and a.compare["Rangbaseline"]["shd"] == 0 and a.compare["PC"]["undirected"] == 2 and a.compare["PC"]["correct"] == 3 and a.compare["PC"]["shd"] == 1.0
    assert a.varsort == 1.0 and a.h < 1e-3 and len(a.trace) >= 5 and a.steps > 100 and len(a.pairs) == 5 and all(r["prefers"] == "a → b" for r in a.pairs)
    assert a.variances == pytest.approx([1.0, 1.64, 2.05, 3.63, 1.0], abs=0.25)
    assert ev.analyse(ev.Settings()).learned == a.learned


def test_standardized_analysis_has_ties_and_wrong_directions():
    a = ev.analyse(ev.Settings(scale="standardisiert"))
    assert a.varsort == 0.5 and all(r["prefers"] == "Gleichstand" for r in a.pairs) and a.variances == pytest.approx([1.0] * 5)
    assert a.compare["NOTEARS"]["reversed"] >= 3 and a.compare["NOTEARS"]["correct"] <= 1 and a.compare["PC"]["shd"] == 1.0                 # PC ist skalenfrei
    assert ev.analyse(ev.Settings()).compare["PC"] == a.compare["PC"]


def test_pairs_prefer_the_reverse_when_the_root_has_more_variance():
    a = ev.analyse(ev.Settings(sigma=2.0))
    assert [r["prefers"] for r in a.pairs[:2]] == ["b → a", "b → a"] and (W.S, W.Z) not in a.learned


def test_pair_rows_obey_the_closed_form():
    a = ev.analyse(ev.Settings(sigma=1.5, seed=7))
    for r in a.pairs:
        assert r["difference"] == pytest.approx(0.5 * r["rho2"] * (r["var_i"] - r["var_j"]), abs=1e-12) and r["loss_forward"] > 0


def test_penalty_and_threshold_remove_the_weak_edge():
    for kw in (dict(lam=0.3), dict(thr=0.5)):
        a = ev.analyse(ev.Settings(**kw))
        assert (W.W, W.Y) not in a.learned and a.compare["NOTEARS"]["missing"] == 1 and a.compare["Rangbaseline"]["shd"] == 0
    assert ev.analyse(ev.Settings(lam=0.3)).w[W.W, W.Y] == pytest.approx(0.28, abs=0.03)


def test_random_experiment_structure(small_random):
    r = small_random
    assert r["nodes"] == 6 and r["n"] == 500 and 1 <= r["dags"] <= 3 and set(r["rows"]) == set(C.SCALES) and set(r["varsort"]) == set(C.SCALES)
    for sc in C.SCALES:
        for m in ev.METHODS:
            x = r["rows"][sc][m]
            assert x["correct"] + x["reversed"] + x["undirected"] <= 1.0 + 1e-12 and x["shd"] >= 0 and x["missing"] >= 0 and x["extra"] >= 0
    assert r["varsort"][C.SCALES[0]] > 0.8 and r["varsort"][C.SCALES[1]] == pytest.approx(0.5, abs=0.2)
    assert r["rows"][C.SCALES[0]]["PC"] == r["rows"][C.SCALES[1]]["PC"]                                       # PC ändert sich mit der Skala nicht


def test_world_experiment_structure(small_world):
    r = small_world
    assert r["reps"] == 2 and [(x["n"], x["scale"]) for x in r["rows"]] == [(300, "roh"), (300, "standardisiert")] and r["varsort_raw"] == 1.0 and r["varsort_std"] == 0.5
    for x in r["rows"]:
        assert 0 <= x["full"] <= x["sz"] + 1e-12 + 1 and x["full"] * 2 == pytest.approx(round(x["full"] * 2)) and set(x["methods"]) == set(ev.METHODS)
    raw, std = r["rows"]
    assert raw["methods"]["PC"] == std["methods"]["PC"] and raw["methods"]["PC"]["undirected"] > 0.3


def test_sigma_experiment_structure_and_variances(small_sigma):
    r = small_sigma
    assert r["reps"] == 3 and r["n"] == 500 and r["flip"] == pytest.approx(1 / 0.6) and [x["sigma"] for x in r["rows"]] == [1.0, 1.5, 2.0]
    for x in r["rows"]:
        assert x["var_s_exact"] == x["sigma"] ** 2 and x["var_z_exact"] == pytest.approx(0.64 * x["sigma"] ** 2 + 1.0) and x["var_s"] == pytest.approx(x["var_s_exact"], rel=0.25)
        assert 0 <= x["notears"] <= 1 and 0 <= x["rank"] <= 1
    assert r["rows"][0]["rank"] == 1.0 and r["rows"][-1]["rank"] < r["rows"][0]["rank"] and r["rows"][-1]["var_s"] > r["rows"][-1]["var_z"]


def test_constraint_experiment_structure(small_constraint):
    r = small_constraint
    assert r["reps_unconstrained"] == 3 and r["reps_grid"] == 2 and r["cyclic"] == 1.0 and r["h_final"] < 1e-3 and len(r["trace"]) >= 5
    assert [(g["lam"], g["thr"]) for g in r["grid"]] == [(0.1, 0.3), (0.1, 0.5), (0.3, 0.3), (0.3, 0.5)]
    grid = {(g["lam"], g["thr"]): g for g in r["grid"]}
    assert grid[(0.1, 0.3)]["shd"] == 0 and grid[(0.1, 0.3)]["exact"] == 1.0 and grid[(0.3, 0.5)]["shd"] > grid[(0.1, 0.3)]["shd"] and grid[(0.1, 0.5)]["missing"] >= grid[(0.1, 0.3)]["missing"]


def test_builder_by_hand():
    two = ev.builder_result([(0, 1), (1, 0)], 1.0)
    assert two["h"] == pytest.approx(1.0862, abs=5e-5) and not two["acyclic"] and two["edges"] == 2
    three = ev.builder_result([(0, 1), (1, 2), (2, 0)], 1.0)
    assert three["h"] == pytest.approx(0.5042, abs=5e-5) and not three["acyclic"]
    chain = ev.builder_result([(0, 1), (1, 2)], 1.5)
    assert chain["h"] == pytest.approx(0.0, abs=1e-12) and chain["acyclic"] and chain["w"][0, 1] == 1.5 and np.count_nonzero(chain["w"]) == 2
    assert ev.builder_result([], 1.0)["h"] == 0.0 and ev.BUILDER_NODES == ("A", "B", "C") and len(ev.BUILDER_ARROWS) == 6
    heavy = ev.builder_result([(0, 1), (1, 0)], 2.0)
    assert heavy["h"] == pytest.approx(2 * np.cosh(4.0) - 2) and heavy["h"] > two["h"]


def test_builder_h_matches_acyclicity_for_every_subset():
    import itertools

    for k in range(7):
        for sel in itertools.combinations(ev.BUILDER_ARROWS, k):
            b = ev.builder_result(sel, 1.0)
            assert (b["h"] < 1e-9) == b["acyclic"] == N.is_acyclic(set(sel)), sel
