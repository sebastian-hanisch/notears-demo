"""Presets, Reglergrenzen, Permalink-Raster und Konstanten."""

import pytest

import nts_constants as C
import nts_evaluation as ev
import nts_presets as P
import stl_world as W


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            v = p[key]
            if spec.choices:
                assert v in spec.choices, (name, key)
                continue
            assert spec.lo <= v <= spec.hi, (name, key)
            assert P.snap(v, spec) == pytest.approx(v), (name, key)


def test_first_preset_is_the_default_setting():
    p = C.PRESETS["Rohdaten (Voreinstellung)"]
    assert ev.Settings() == ev.Settings(**p) and next(iter(C.PRESETS)) == "Rohdaten (Voreinstellung)"
    assert all(P.SETTING_SPECS[sk].default == p[k] for k, sk in P.PRESET_KEYS.items())


def test_presets_differ_and_each_changes_at_most_two_settings():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)
    base = C.PRESETS["Rohdaten (Voreinstellung)"]
    for name, p in C.PRESETS.items():
        assert len([k for k in p if p[k] != base[k]]) <= 2, name
    assert C.PRESETS["Standardisiert"]["scale"] == "standardisiert" and C.PRESETS["Wurzel knapp"]["sigma"] == 1.5 and C.PRESETS["Wurzel streut stark"]["sigma"] == 2.0 and C.PRESETS["Wenige Standorte"]["n"] == 100
    assert C.PRESETS["Starke Strafe"]["lam"] == 0.3 and C.PRESETS["Hohe Schwelle"]["thr"] == 0.5


def test_the_presets_show_what_their_help_says():
    by = {n: ev.analyse(ev.Settings(**p)) for n, p in C.PRESETS.items()}
    base = by["Rohdaten (Voreinstellung)"]
    assert base.learned == base.truth and base.compare["PC"]["undirected"] == 2
    std = by["Standardisiert"]
    assert std.compare["NOTEARS"]["correct"] == 0 and std.compare["NOTEARS"]["reversed"] >= 3 and std.compare["PC"] == base.compare["PC"]
    knapp = by["Wurzel knapp"]
    assert knapp.rank == knapp.truth and knapp.compare["NOTEARS"]["shd"] > 0 and knapp.variances[W.S] == pytest.approx(2.25, abs=0.4) and knapp.variances[W.S] < knapp.variances[W.Z]
    streut = by["Wurzel streut stark"]
    assert streut.variances[W.S] > streut.variances[W.Z] > streut.variances[W.X] and (W.S, W.Z) not in streut.learned and (W.Z, W.X) not in streut.learned
    wenige = by["Wenige Standorte"]
    assert wenige.compare["NOTEARS"]["shd"] > 0 and (W.Z, W.Y) not in wenige.learned and any(a == W.Z and b == W.Y and st == "fehlt" for a, b, st in wenige.status)
    assert (W.W, W.Y) not in by["Starke Strafe"].learned and abs(by["Starke Strafe"].w[W.W, W.Y]) < 0.3
    assert (W.W, W.Y) not in by["Hohe Schwelle"].learned and 0.4 < by["Hohe Schwelle"].w[W.W, W.Y] < 0.5


def test_snap_clamps_rounds_and_maps_the_choices():
    sg = P.SETTING_SPECS["sigma_slider"]
    assert P.snap(9.0, sg) == C.SIGMA_MAX and P.snap(0.0, sg) == C.SIGMA_MIN and P.snap(1.3, sg) == pytest.approx(1.25) and P.snap(1.4, sg) == pytest.approx(1.5)
    n = P.SETTING_SPECS["n_slider"]
    assert P.snap(50, n) == C.N_MIN and P.snap(10 ** 6, n) == C.N_MAX and P.snap(1049, n) == 1000 and P.snap(1051, n) == 1100
    th = P.SETTING_SPECS["thr_slider"]
    assert P.snap(0.33, th) == pytest.approx(0.35) and P.snap(0.32, th) == pytest.approx(0.3)
    lam = P.SETTING_SPECS["lam_slider"]
    assert P.snap(0, lam) == 0.01 and P.snap(2, lam) == 0.3 and P.snap(9, lam) == 0.3 and P.snap(-1, lam) == 0.01 and P.url_value("lam_slider", 0.1) == 1 and P.url_value("sigma_slider", 1.5) == 1.5
    sc = P.SETTING_SPECS["scale_radio"]
    assert P.snap(1, sc) == "standardisiert" and P.snap(0, sc) == "roh"


def test_bounds_defaults_and_params():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("thr_slider") == (C.THR_MIN, C.THR_MAX) and P.bounds("lam_slider") == (0, 2)
    assert {s.url_param for s in P.SETTING_SPECS.values()} == {"sc", "ss", "n", "l", "th", "seed"} and C.DEFAULT_N in range(C.N_MIN, C.N_MAX + 1, C.N_STEP) and C.DEFAULT_LAMBDA in C.LAMBDAS and C.DEFAULT_SCALE == "roh"


def test_experiment_constants_are_consistent():
    assert C.EXP_RANDOM_DAGS >= 10 and C.EXP_RANDOM_NODES == 6 and C.EXP_WORLD_NS == tuple(sorted(C.EXP_WORLD_NS)) and C.EXP_WORLD_REPS >= 10 and C.EXP_SIGMAS == tuple(sorted(C.EXP_SIGMAS)) and 1.0 in C.EXP_SIGMAS
    assert C.SIGMA_FLIP == pytest.approx(1 / 0.6) and C.SIGMA_FLIP ** 2 == pytest.approx(0.64 * C.SIGMA_FLIP ** 2 + 1)
    assert 0.1 in C.EXP_GRID_LAMBDAS and 0.3 in C.EXP_GRID_LAMBDAS and 0.3 in C.EXP_GRID_THRS and C.EXP_UNCONSTRAINED_REPS >= 10 and C.PC_ALPHA == 0.01
