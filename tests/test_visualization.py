"""Figuren und Tabellen: bauen ohne Fehler, und die gezeichneten Zahlen sind die der Auswertung."""

import pytest

import nts_constants as C
import nts_evaluation as ev
import nts_visualization as V
import stl_world as W


def _arrows(fig):
    return [ann for ann in fig.layout.annotations if ann.showarrow]


def test_true_diagram_draws_every_arrow_and_marks_x_and_y():
    fig = V.build_true_diagram(W.World().parents)
    assert len(_arrows(fig)) == 5
    nodes = {t.text[0]: t for t in fig.data if t.mode == "markers+text"}
    assert set(nodes) == set(W.NAMES) and nodes["X"].marker.line.color == C.COLOR_GOOD and nodes["Y"].marker.line.color == C.COLOR_TRUTH


def test_learned_diagram_is_green_when_everything_is_right():
    fig = V.build_learned_diagram(ev.analyse(ev.Settings()))
    assert {ann.arrowcolor for ann in _arrows(fig)} == {C.COLOR_GOOD} and len(_arrows(fig)) == 5 and {t.name for t in fig.data if t.name} == {"richtig"}


def test_learned_diagram_shows_reversed_missing_and_extra_edges():
    a = ev.analyse(ev.Settings(scale="standardisiert"))
    fig = V.build_learned_diagram(a)
    assert C.COLOR_TRUTH in {ann.arrowcolor for ann in _arrows(fig)} and {"umgekehrt", "fehlt"} <= {t.name for t in fig.data if t.name}
    assert any(t.line.dash == "dash" for t in fig.data if t.mode == "lines" and t.line.color == C.COLOR_NEUTRAL)
    assert len(_arrows(fig)) == len([e for e in a.status if e[2] != "fehlt"])


def test_edge_table_by_hand():
    a = ev.analyse(ev.Settings())
    t = V.build_edge_table(a)
    assert list(t.columns) == ["Kante", "Urteil", "Gewicht W (vor der Schwelle)", "Gegenrichtung"] and len(t) == 5
    row = t[t["Kante"] == "S → Z"].iloc[0]
    assert row["Urteil"] == "richtig" and row["Gewicht W (vor der Schwelle)"] == ev.signed(a.w[0, 1], 2) and float(row["Gewicht W (vor der Schwelle)"].replace("+", "").replace(",", ".")) == pytest.approx(0.65, abs=0.08)
    bad = V.build_edge_table(ev.analyse(ev.Settings(lam=0.3)))
    assert (bad["Urteil"] == "fehlt").sum() == 1
    V.style_edge_table(bad)


def test_methods_and_summary_tables_by_hand():
    a = ev.analyse(ev.Settings())
    t = V.build_methods_table(a)
    assert list(t["Verfahren"]) == ["NOTEARS", "Rangbaseline", "PC"] and t.iloc[0]["richtig gerichtet"] == "5 von 5" and t.iloc[2]["ungerichtet"] == 2 and t.iloc[2]["SHD"] == "1,0"
    s = V.build_summary_table(a)
    assert len(s) == 5 and s.iloc[0]["Wert"] == "1,00" and s.iloc[1]["Wert"] == ", ".join(ev.de(v, 2) for v in a.variances) and s.iloc[4]["Wert"] == "1000, 0,10, 0,30"


def test_pair_chart_and_table_agree():
    a = ev.analyse(ev.Settings())
    fig = V.build_pair_chart(a)
    assert [t.name.split(" ")[1] for t in fig.data] == ["Ursache", "Wirkung"] and list(fig.data[0].y) == pytest.approx([r["loss_forward"] for r in a.pairs]) and list(fig.data[1].y) == pytest.approx([r["loss_backward"] for r in a.pairs])
    t = V.build_pair_table(a)
    assert len(t) == 5 and set(t["bevorzugt"]) == {"a → b"} and list(t["Differenz"]) == list(t["½ ρ² (Var Ursache − Var Wirkung)"])
    assert set(V.build_pair_table(ev.analyse(ev.Settings(scale="standardisiert")))["bevorzugt"]) == {"Gleichstand"}


def test_trace_chart_and_table():
    a = ev.analyse(ev.Settings())
    fig = V.build_trace_chart(a.trace)
    assert len(fig.data) == 2 and list(fig.data[1].y) == [t["rho"] for t in a.trace] and fig.layout.yaxis.type == "log"
    t = V.build_trace_table(a.trace)
    assert len(t) == len(a.trace) and list(t["Schritt"]) == list(range(1, len(a.trace) + 1)) and t.iloc[0]["ρ"] == "1e+00"


def test_experiment_figures_and_tables_build(small_random, small_world, small_sigma, small_constraint):
    assert V.build_random_chart(small_random) is not None and len(V.build_random_table(small_random)) == 6
    assert V.build_world_chart(small_world) is not None and len(V.build_world_table(small_world)) == 2
    assert V.build_sigma_chart(small_sigma) is not None and len(V.build_sigma_table(small_sigma)) == 3
    assert len(V.build_grid_table(small_constraint)) == 4


def test_random_chart_stacks_right_and_reversed_shares(small_random):
    fig = V.build_random_chart(small_random)
    assert [t.name for t in fig.data] == ["richtig gerichtet", "umgekehrt"] and len(fig.data[0].x) == 6 and fig.layout.barmode == "stack"
    r = small_random["rows"]["roh"]["NOTEARS"]
    assert fig.data[0].y[0] == pytest.approx(100 * r["correct"]) and fig.data[1].y[0] == pytest.approx(100 * r["reversed"])


def test_sigma_chart_marks_the_flip(small_sigma):
    fig = V.build_sigma_chart(small_sigma)
    assert [t.name for t in fig.data] == ["NOTEARS", "Rangbaseline"] and fig.layout.shapes[0].x0 == pytest.approx(1 / 0.6)


def test_builder_diagram_draws_the_selected_arrows():
    own = V.build_builder_diagram([(0, 1), (1, 2), (2, 0)], False)
    assert len(_arrows(own)) == 3 and {ann.arrowcolor for ann in _arrows(own)} == {C.COLOR_BAD} and len([t for t in own.data if t.mode == "markers+text"]) == 3
    ok = V.build_builder_diagram([(0, 1)], True)
    assert {ann.arrowcolor for ann in _arrows(ok)} == {C.COLOR_GOOD}
    t = V.build_builder_table(ev.builder_result([(0, 1), (1, 2)], 1.5))
    assert list(t["Gewicht"]) == ["1,50", "1,50"] and list(t["von"]) == ["A", "B"] and V.build_builder_table(ev.builder_result([], 1.0)).empty
