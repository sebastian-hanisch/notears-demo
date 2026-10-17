"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py, klickt jeden Preset-Button, fährt die Regler an die Grenzen und prüft, dass kein Python-Fehler auftritt."""

import os

from streamlit.testing.v1 import AppTest

import nts_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SLIDERS = ("sigma_slider", "n_slider", "thr_slider")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_loads_without_exception_and_shows_the_numbers():
    at = _fresh_app()
    m = _metrics(at)
    assert m["NOTEARS: richtig gerichtet"] == "5 von 5" and m["Rangbaseline: richtig gerichtet"] == "5 von 5" and m["PC: richtig gerichtet"] == "3 von 5" and m["Varsortability"] == "1,00"
    assert at.radio(key="scale_radio").value == C.DEFAULT_SCALE and at.slider(key="n_slider").value == C.DEFAULT_N and at.select_slider(key="lam_slider").value == C.DEFAULT_LAMBDA and at.number_input(key="seed_input").value == C.DEFAULT_SEED
    assert any("NOTEARS trifft das wahre Diagramm genau" in s.value for s in at.success)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if label.startswith("▶"):
            continue                                                         # die Experimente laufen in eigenen Tests
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_the_sliders_at_min_and_max_do_not_raise():
    for key in SLIDERS:
        for edge in ("min", "max"):
            at = _fresh_app()
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.radio(key="scale_radio").value == preset["scale"] and at.slider(key="sigma_slider").value == preset["sigma"] and at.slider(key="n_slider").value == preset["n"]
        assert at.select_slider(key="lam_slider").value == preset["lam"] and at.slider(key="thr_slider").value == preset["thr"] and at.number_input(key="seed_input").value == preset["seed"]


def test_standardized_preset_shows_the_warning_and_the_wrong_directions():
    at = _fresh_app()
    at.button[[b.label for b in at.button].index("Standardisiert")].click().run(timeout=TIMEOUT)
    assert not at.exception and any("Nach der Standardisierung haben alle Spalten Varianz 1" in w.value for w in at.warning)
    m = _metrics(at)
    assert m["NOTEARS: richtig gerichtet"].startswith("0 von 5") and m["Varsortability"] == "0,50" and m["PC: richtig gerichtet"] == "3 von 5"


def test_scale_radio_and_lambda_work():
    at = _fresh_app()
    at.radio(key="scale_radio").set_value(C.SCALES[1]).run(timeout=TIMEOUT)
    at.select_slider(key="lam_slider").set_value(0.3).run(timeout=TIMEOUT)
    assert not at.exception and at.radio(key="scale_radio").value == "standardisiert" and at.select_slider(key="lam_slider").value == 0.3


def test_builder_default_examples_and_toggle():
    at = _fresh_app()
    assert at.checkbox(key="hb_AB").value and at.checkbox(key="hb_BC").value and at.checkbox(key="hb_CA").value and not at.checkbox(key="hb_BA").value
    assert _metrics(at)["h(W)"] == "0,5042" and any("gerichteter Kreis" in e.value for e in at.error)
    labels = [b.label for b in at.button]
    for name, expected in (("Zwei-Zyklus A ⇄ B", "1,0862"), ("Kette A → B → C", "0,0000"), ("Dreieck ohne Kreis", "0,0000"), ("Drei-Zyklus A → B → C → A", "0,5042")):
        at.button[labels.index(name)].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert _metrics(at)["h(W)"] == expected, name
    at.checkbox(key="hb_CA").set_value(False).run(timeout=TIMEOUT)
    assert not at.exception and _metrics(at)["h(W)"] == "0,0000"


def test_every_experiment_runs_on_click():
    at = _fresh_app()
    assert "exp1_ready" not in at.session_state
    for key, text in (("exp1_run", "Zufalls-DAGs mit 6 Knoten"), ("exp2_run", "In der Welt der Linie"), ("exp3_run", "Die Varianzordnung von Saison und Nachfrage kippt"), ("exp4_run", "Das augmentierte Lagrange-Verfahren braucht")):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
        assert any(text in i.value for i in at.info), key


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_taken_and_clamped():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "50"
    at.query_params["ss"] = "9"
    at.query_params["sc"] = "1"
    at.query_params["l"] = "2"
    at.query_params["seed"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="sigma_slider").value == C.SIGMA_MAX and at.radio(key="scale_radio").value == "standardisiert"
    assert at.select_slider(key="lam_slider").value == 0.3 and at.number_input(key="seed_input").value == C.DEFAULT_SEED


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="sigma_slider").set_value(1.5).run(timeout=TIMEOUT)
    assert at.query_params["ss"] in ("1.5", ["1.5"])
    at.radio(key="scale_radio").set_value(C.SCALES[1]).run(timeout=TIMEOUT)
    assert at.query_params["sc"] in ("1", ["1"])
    at.select_slider(key="lam_slider").set_value(0.01).run(timeout=TIMEOUT)
    assert at.query_params["l"] in ("0", ["0"])
