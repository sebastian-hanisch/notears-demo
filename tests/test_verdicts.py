"""Verdikte: Art und Text je Lage; jede Zahl im Text stammt aus dem Befund, nicht aus einer Konstanten."""

import nts_evaluation as ev


def test_instance_verdict_default_is_a_success():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings()))
    assert kind == "success" and "Roh: NOTEARS richtet 5 von 5 wahren Kanten richtig und 0 falsch herum (SHD 0,0)" in text and "PC lässt 2 ungerichtet und richtet 3 richtig (SHD 1,0)" in text
    assert "Varsortability der Daten: 1,00" in text and "NOTEARS trifft das wahre Diagramm genau" in text


def test_instance_verdict_standardized_names_the_scale():
    a = ev.analyse(ev.Settings(scale="standardisiert"))
    kind, text = ev.verdict_instance(a)
    assert kind == "warning" and text.startswith("Standardisiert: NOTEARS richtet") and "Nach der Standardisierung haben alle Spalten Varianz 1" in text and "umgekehrt" in text and "Varsortability der Daten: 0,50" in text


def test_instance_verdict_lists_the_deviations_when_raw():
    kind, text = ev.verdict_instance(ev.analyse(ev.Settings(lam=0.3)))
    assert kind == "warning" and "Abweichungen: W → Y fehlt." in text and "Nach der Standardisierung" not in text and "richtet 4 von 5" in text


def test_pairs_verdict_counts_the_directions():
    kind, text = ev.verdict_pairs(ev.analyse(ev.Settings()))
    assert kind == "info" and "5 von 5 bevorzugen die richtige, 0 die falsche, 0 sind ein Gleichstand" in text and "½ ρ² (Var(Ursache) − Var(Wirkung))" in text
    text2 = ev.verdict_pairs(ev.analyse(ev.Settings(scale="standardisiert")))[1]
    assert "0 von 5 bevorzugen die richtige, 0 die falsche, 5 sind ein Gleichstand" in text2
    text3 = ev.verdict_pairs(ev.analyse(ev.Settings(sigma=2.0)))[1]
    assert "2 die falsche" in text3 and "3 von 5 bevorzugen die richtige" in text3


def test_random_verdict(small_random):
    kind, text = ev.verdict_random(small_random)
    r, s = small_random["rows"]["roh"], small_random["rows"]["standardisiert"]
    assert kind == "info" and f"In {small_random['dags']} Zufalls-DAGs mit 6 Knoten (n = 500)" in text and f"NOTEARS richtet roh {ev.pct(r['NOTEARS']['correct'])}" in text and f"standardisiert sind es {ev.pct(s['NOTEARS']['correct'])}" in text
    assert f"PC ändert sich nicht (SHD {ev.de(r['PC']['shd'], 2)})" in text


def test_world_verdict(small_world):
    kind, text = ev.verdict_world(small_world)
    raw = small_world["rows"][0]
    assert kind == "info" and "(n = 300, 2 Läufe je Zelle)" in text and f"S → Z in {ev.pct(raw['sz'])}" in text and "PC lässt beide Kanten offen" in text and "Mehr Standorte helfen nicht" in text


def test_sigma_verdict(small_sigma):
    kind, text = ev.verdict_sigma(small_sigma)
    mid, last = small_sigma["rows"][1], small_sigma["rows"][2]
    assert kind == "info" and "kippt bei σ = 1,67" in text and f"Bei σ = 1,5 (Var(S) {ev.de(mid['var_s'], 2)} gegen Var(Z) {ev.de(mid['var_z'], 2)})" in text and f"bei σ = 2,0 richtet NOTEARS sie in {ev.pct(last['notears'])}, die Rangbaseline in {ev.pct(last['rank'])} richtig" in text


def test_constraint_verdict(small_constraint):
    kind, text = ev.verdict_constraint(small_constraint)
    tr = small_constraint["trace"]
    assert kind == "info" and f"braucht {len(tr)} Teilprobleme" in text and f"von {ev.de(tr[0]['h'], 3)} auf" in text and "Ohne sie ist der Graph in 100 % der 3 Läufe zyklisch" in text and "bei λ = 0,1 trifft die Schwelle 0,3 (SHD 0,00)" in text


def test_builder_verdicts():
    cyc = ev.verdict_builder(ev.builder_result([(0, 1), (1, 2), (2, 0)], 1.0))
    assert cyc[0] == "error" and "gerichteter Kreis: h(W) = 0,5042" in cyc[1]
    ok = ev.verdict_builder(ev.builder_result([(0, 1), (1, 2)], 1.0))
    assert ok[0] == "success" and "Kein gerichteter Kreis: h(W) = 0,0000" in ok[1]
    empty = ev.verdict_builder(ev.builder_result([], 1.0))
    assert empty[0] == "info" and "h(W) = 0" in empty[1]
