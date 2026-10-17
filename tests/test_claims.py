"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über SplitMix64-Ströme (ganzzahlig, plattformstabil); Fließkomma-Reihenfolgen und der Löser können je Bibliotheksversion die letzte Stelle ändern, deshalb Toleranzen von einer halben
Einheit der gedruckten Stelle. Was Konstruktion ist, steht ausdrücklich dabei: dass die Varsortability standardisierter Daten 0,5 ist (alle Varianzen gleich), dass ohne Nebenbedingung Zyklen entstehen
(kein Zyklenverbot im Modell) und dass der Verlust zweier Richtungen exakt ½ ρ² (Var₁ − Var₂) ist (Rechenidentität).
"""

import math

import numpy as np
import pytest

import nts_baselines as B
import nts_constants as C
import nts_evaluation as ev
import nts_notears as N
import stl_world as W


def test_default_sample():
    a = ev.analyse(ev.Settings())
    assert a.learned == a.truth and a.rank == a.truth and a.compare["PC"]["correct"] == 3 and a.compare["PC"]["undirected"] == 2 and a.compare["PC"]["shd"] == 1.0 and a.varsort == 1.0
    assert ev.verdict_instance(a)[0] == "success" and "Roh: NOTEARS richtet 5 von 5 wahren Kanten richtig und 0 falsch herum (SHD 0,0)" in ev.verdict_instance(a)[1]
    std = ev.analyse(ev.Settings(scale="standardisiert"))
    assert std.compare["NOTEARS"]["correct"] == 0 and std.compare["NOTEARS"]["reversed"] == 4 and std.compare["NOTEARS"]["missing"] == 1 and std.compare["NOTEARS"]["shd"] == 5.0 and std.compare["Rangbaseline"]["shd"] == 6.0


def test_hand_values_of_the_text():
    assert N.h_value(np.array([[0, 1.0], [1.0, 0]])) == pytest.approx(1.0862, abs=0.00006) and N.h_value(np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])) == pytest.approx(0.5042, abs=0.00006)
    assert N.h_value(np.array([[0, 1.0], [0, 0]])) == pytest.approx(0.0, abs=1e-12)
    r = B.pair_losses(np.array([[1.0, 0.8], [0.8, 1.64]]), 0, 1)
    assert r["loss_forward"] == pytest.approx(1.000, abs=0.0006) and r["loss_backward"] == pytest.approx(1.1249, abs=0.00006) and r["difference"] == pytest.approx(-0.1249, abs=0.00006) and r["rho2"] == pytest.approx(0.3902, abs=0.00006)
    tie = B.pair_losses(np.array([[1.0, 0.8], [0.8, 1.0]]), 0, 1)
    assert tie["difference"] == pytest.approx(0.0, abs=1e-15) and tie["loss_forward"] == pytest.approx(0.5 * (2 - 0.64) + 0.0)
    cov = W.population_cov(W.World().matrix)
    assert [round(cov[i, i], 4) for i in range(5)] == [1.0, 1.64, 2.0496, 3.6329, 1.0]
    assert 1 / 0.6 == pytest.approx(1.6667, abs=0.00006) and math.isclose((1 / 0.6) ** 2, 0.64 * (1 / 0.6) ** 2 + 1)
    assert math.cosh(1) * 2 - 2 == pytest.approx(1.0862, abs=0.00006) and math.e + 2 * math.exp(-0.5) * math.cos(math.sqrt(3) / 2) - 3 == pytest.approx(0.5042, abs=0.00006)


def test_closed_form_is_an_identity_not_a_finding():
    """Verlust(1 → 2) − Verlust(2 → 1) = ½ ρ² (Var₁ − Var₂) gilt für jede Kovarianz (Rechenidentität); der Befund ist, was daraus für Standardisierung und Rauschstärke folgt."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        a = rng.normal(size=(2, 2))
        cov = a @ a.T + 0.05 * np.eye(2)
        r = B.pair_losses(cov, 0, 1)
        assert r["difference"] == pytest.approx(0.5 * r["rho2"] * (cov[0, 0] - cov[1, 1]), abs=1e-12)


def test_experiment_1_random_dags(full_random):
    r = full_random
    assert r["dags"] == 20 and r["nodes"] == 6 and r["n"] == 1000 and C.EXP_RANDOM_DAGS == 20
    assert r["varsort"]["roh"] == pytest.approx(0.9605, abs=0.0006) and r["varsort"]["standardisiert"] == 0.5                  # standardisiert: Konstruktion (alle Varianzen gleich)
    raw, std = r["rows"]["roh"], r["rows"]["standardisiert"]
    assert (raw["NOTEARS"]["correct"], raw["NOTEARS"]["reversed"], raw["NOTEARS"]["shd"]) == pytest.approx((0.935, 0.0244, 0.4), abs=0.0006)
    assert (raw["Rangbaseline"]["correct"], raw["Rangbaseline"]["reversed"], raw["Rangbaseline"]["shd"]) == pytest.approx((0.9268, 0.065, 0.75), abs=0.0006)
    # standardisiert entscheidet Rundungsrauschen über die Gleichstände des Verlusts: NOTEARS weicht je Plattform um eine Kante ab (CI: 0,317 / 0,366 / 4,65), die Rangbaseline ist über die feste Zufallsreihenfolge bei Gleichstand reproduzierbar
    assert (std["NOTEARS"]["correct"], std["NOTEARS"]["reversed"], std["NOTEARS"]["shd"]) == pytest.approx((0.32, 0.36, 4.6), abs=0.02) and std["NOTEARS"]["shd"] == pytest.approx(4.6, abs=0.1)
    assert (std["Rangbaseline"]["correct"], std["Rangbaseline"]["reversed"], std["Rangbaseline"]["shd"]) == pytest.approx((0.4553, 0.4797, 6.0), abs=0.0006)
    assert raw["PC"] == std["PC"] and (raw["PC"]["correct"], raw["PC"]["reversed"], raw["PC"]["undirected"], raw["PC"]["shd"]) == pytest.approx((0.3984, 0.1951, 0.2764, 2.9), abs=0.0006)
    assert raw["Rangbaseline"]["extra"] == pytest.approx(0.3, abs=0.0006) and raw["NOTEARS"]["extra"] == 0.0 and raw["NOTEARS"]["missing"] == pytest.approx(0.25, abs=0.0006)
    text = ev.verdict_random(r)[1]
    assert "In 20 Zufalls-DAGs mit 6 Knoten (n = 1 000) fällt die Varsortability von 0,96 auf 0,50" in text and "NOTEARS richtet roh 93 % der wahren Kanten richtig (SHD 0,40), die Rangbaseline 93 % (SHD 0,75)" in text and "PC ändert sich nicht (SHD 2,90)" in text


def test_experiment_2_the_world(full_world):
    r = full_world
    rows = {(x["n"], x["scale"]): x for x in r["rows"]}
    assert r["reps"] == 16 and C.EXP_WORLD_REPS == 16 and r["varsort_raw"] == 1.0 and r["varsort_std"] == 0.5
    raw = rows[(1000, "roh")]
    assert (raw["sz"], raw["zx"], raw["full"]) == (1.0, 1.0, 1.0) and raw["methods"]["NOTEARS"]["shd"] == 0.0 and raw["methods"]["Rangbaseline"]["shd"] == 0.0
    assert (raw["methods"]["PC"]["correct"], raw["methods"]["PC"]["undirected"], raw["methods"]["PC"]["shd"]) == (0.6, 0.4, 1.0)
    small = rows[(300, "roh")]
    assert (small["sz"], small["zx"], small["full"]) == (1.0, 0.9375, 0.875) and small["methods"]["NOTEARS"]["correct"] == pytest.approx(0.9625) and small["methods"]["NOTEARS"]["reversed"] == pytest.approx(0.025)
    for n, correct, reversed_, missing, shd in ((300, 0.05, 0.7875, 0.8125, 4.75), (1000, 0.025, 0.7375, 1.1875, 4.875)):
        s = rows[(n, "standardisiert")]
        assert (s["sz"], s["zx"], s["full"]) == (0.0, 0.0, 0.0)
        assert (s["methods"]["NOTEARS"]["correct"], s["methods"]["NOTEARS"]["reversed"], s["methods"]["NOTEARS"]["missing"], s["methods"]["NOTEARS"]["shd"]) == pytest.approx((correct, reversed_, missing, shd), abs=0.0006)
        assert s["methods"]["PC"] == rows[(n, "roh")]["methods"]["PC"]
    assert (rows[(1000, "standardisiert")]["methods"]["Rangbaseline"]["correct"], rows[(1000, "standardisiert")]["methods"]["Rangbaseline"]["extra"]) == pytest.approx((0.4125, 2.125), abs=0.0006)
    text = ev.verdict_world(r)[1]
    assert "(n = 1 000, 16 Läufe je Zelle)" in text and "S → Z in 100 % und Z → X in 100 %" in text and "(ungerichtet 40 % der wahren Kanten)" in text and "Standardisiert sind es 0 %, 0 % und 0 %, 74 % der Kanten sind umgekehrt" in text


def test_experiment_3_noise_of_the_root(full_sigma):
    r = full_sigma
    rows = {x["sigma"]: x for x in r["rows"]}
    assert r["reps"] == 20 and r["n"] == 1000 and r["flip"] == pytest.approx(1.6667, abs=0.0006)
    assert (rows[1.0]["notears"], rows[1.0]["rank"]) == (1.0, 1.0) and (rows[1.5]["notears"], rows[1.5]["rank"]) == (0.4, 1.0) and (rows[2.0]["notears"], rows[2.0]["rank"]) == (0.0, 0.0)
    assert (rows[1.5]["var_s"], rows[1.5]["var_z"]) == pytest.approx((2.2453, 2.4541), abs=0.0006) and (rows[2.0]["var_s"], rows[2.0]["var_z"]) == pytest.approx((4.0082, 3.4882), abs=0.0006)
    assert (rows[1.5]["var_s_exact"], rows[1.5]["var_z_exact"], rows[2.0]["var_z_exact"]) == pytest.approx((2.25, 2.44, 3.56))              # Var(Z) = 0,64 σ² + 1
    text = ev.verdict_sigma(r)[1]
    assert "kippt bei σ = 1,67" in text and "Bei σ = 1,5 (Var(S) 2,25 gegen Var(Z) 2,45) richtet NOTEARS S → Z in 40 % der Läufe richtig, die Rangbaseline in 100 %" in text and "bei σ = 2,0 richtet NOTEARS sie in 0 %, die Rangbaseline in 0 % richtig" in text


def test_experiment_4_constraint(full_constraint):
    r = full_constraint
    tr = r["trace"]
    assert len(tr) == 14 and [t["rho"] for t in tr] == [1, 1, 10, 100, 1000, 1000, 1e4, 1e4, 1e5, 1e5, 1e6, 1e6, 1e7, 1e7]
    assert (tr[0]["h"], tr[0]["loss"], tr[-1]["loss"]) == pytest.approx((0.1116, 1.7961, 2.4443), abs=0.0006) and tr[-1]["h"] == pytest.approx(1.3e-5, abs=1e-6) and (tr[0]["edges"], tr[-1]["edges"]) == (6, 5)
    assert tr[-1]["alpha"] == pytest.approx(419.86, abs=0.05) and r["h_final"] == pytest.approx(1.33e-5, abs=1e-6)
    assert r["cyclic"] == 1.0 and r["reps_unconstrained"] == 30                                                                   # Setup: ohne Zyklenverbot im Modell wählt die Regression beide Richtungen
    grid = {(g["lam"], g["thr"]): g for g in r["grid"]}
    expected = {(0.01, 0.1): (0.5, 0.75), (0.01, 0.3): (0.375, 0.75), (0.01, 0.5): (0.5, 0.75), (0.1, 0.1): (0.0, 1.0), (0.1, 0.3): (0.0, 1.0), (0.1, 0.5): (0.5625, 0.5), (0.3, 0.1): (0.0, 1.0), (0.3, 0.3): (0.3125, 0.6875), (0.3, 0.5): (2.6875, 0.0)}
    for key, (shd, exact) in expected.items():
        assert (grid[key]["shd"], grid[key]["exact"]) == pytest.approx((shd, exact), abs=0.0006), key
    text = ev.verdict_constraint(r)[1]
    assert "braucht 14 Teilprobleme" in text and "von 0,112 auf 1,3·10⁻⁵" in text and "von 1,796 auf 2,444" in text and "Ohne sie ist der Graph in 100 % der 30 Läufe zyklisch" in text and "(SHD 0,00), bei λ = 0,3 nicht mehr (SHD 0,31)" in text


def test_the_solver_is_the_reference_in_the_claims_world():
    """Die Behauptungen hängen am eigenen Löser; auf der Welt der Voreinstellung gibt die scipy-Fassung dieselbe Kantenmenge (test_oracle_notears prüft das breiter)."""
    from test_oracle_notears import reference_notears

    from kausal_generator import SplitMix64

    x = ev.simulate_scaled(W.World(), 1000, SplitMix64(35))
    assert N.threshold_dag(reference_notears(x), 0.3) == ev.truth_edges(W.World())
