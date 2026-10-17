# Kontinuierliches Strukturlernen: Was NOTEARS wirklich sortiert – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-notears-demo.streamlit.app/)**

Vierzehntes Stück der Linie **Kausale Diagramme und Adjustierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Nachfolger von [strukturlernen-demo](https://github.com/sebastian-hanisch/strukturlernen-demo): Dort lässt PC Kanten ungerichtet, weil Unabhängigkeiten sie nicht unterscheiden. Hier die Antwort, die man sich vom Optimieren erhofft, und was sie kostet.

## Kernfrage

**NOTEARS** fragt anders als PC: Welche Gewichtsmatrix **W** erklärt die Daten am besten (Kleinste Quadrate, dazu eine L1-Strafe), **unter der Gleichung h(W) = tr(exp(W∘W)) − d = 0**, die genau für azyklische Graphen gilt? Gelöst wird das mit einem augmentierten Lagrange-Verfahren. Das Ergebnis richtet auf Rohdaten alle Kanten, auch die, die PC offen lässt. Aber die Richtungen stammen aus der **Varianzordnung** der Daten: Eine Standardisierung löscht sie, und eine **Rangbaseline**, die nur Varianzen sortiert, macht roh dasselbe.

## Modell

- **Welt:** die der Linie (Stück 13): Saison S → Nachfrage Z → hoher Faktor X → Servicegrad Y, dazu Z → Y und Wetter W → Y, linear, Einheitsrauschen, Pfeilstärken 0,8 / 0,8 / 0,6 / 0,6 / 0,6; `stl_world.py` und `stl_pc.py` wortgleich aus Stück 13. Regler: Skala (**roh / standardisiert**), Rauschstärke σ der Saison, Standorte, λ, Schwelle, Seed.
- **Zufalls-DAGs** (Experiment 1): sechs Knoten, Kantenwahrscheinlichkeit 0,4, Gewichte ±U(0,5; 1,5), Einheitsrauschen.

## Methodik

- **NOTEARS** (`nts_notears.py`, von Grund auf in numpy): Minimiere ½n ‖X − XW‖² + λ‖W‖₁ + α h(W) + ρ/2 h(W)²; die äußere Schleife erhöht ρ um den Faktor 10, bis h auf höchstens ein Viertel fällt, und setzt α ← α + ρ h; Abbruch bei h ≤ 10⁻⁵ oder ρ ≥ 10⁸. Die Teilprobleme löst ein **proximaler Gradient mit Beschleunigung (FISTA)** mit Rückverfolgung (die Originalarbeit nutzt L-BFGS-B). Schwelle 0,3 auf |W|; bei einem Kreis fällt die schwächste Kante, bis der Graph azyklisch ist. Eigene `expm` (Skalieren und Quadrieren, Taylor).
- **Rangbaseline** (`nts_baselines.py`): Variablen nach wachsender Varianz ordnen, jede auf alle Vorgänger regressieren, Kanten mit |t| < 3 streichen. Keine Strukturschätzung.
- **Varsortability:** Anteil der Paare (Vorfahre, Nachkomme) mit größerer Varianz beim Nachkommen; ein Gleichstand (Toleranz 10⁻⁹) zählt halb.
- **Verlust zweier Richtungen** (geschlossen): Verlust(i → j) − Verlust(j → i) = ½ ρ² (Var i − Var j) aus der Kovarianz; gegen eine zweite Rechnung über Regressionsresiduen geprüft.
- **PC** (stabil, Fisher-z, α = 0,01) aus Stück 13 auf denselben Daten; **SHD** zählt fehlende, überzählige und umgekehrte Kanten je 1, ungerichtete 0,5.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; die Anzahl der festen Wiederholungen steht jeweils dabei (NOTEARS braucht ein bis zwei Sekunden je Anpassung, die Experimente laufen auf Knopfdruck).

| Frage | Befund | Test |
|---|---|---|
| Standard-Stichprobe (n = 1 000, Seed 35, λ = 0,1, Schwelle 0,3), Rohdaten | NOTEARS richtet **5 von 5** wahren Kanten richtig (SHD 0,0), die Rangbaseline ebenso; **PC richtet 3 und lässt 2 ungerichtet** (S–Z und Z–X, SHD 1,0). Varsortability **1,00**. | `test_default_sample` |
| Dieselben Daten standardisiert | NOTEARS richtet **0 von 5** richtig, **4 umgekehrt**, 1 fehlt (SHD **5,0**); die Rangbaseline SHD 6,0; PC unverändert (SHD 1,0). Varsortability **0,50**. | `test_default_sample` |
| Zwei Größen (x₁ ~ N(0, 1), x₂ = 0,8 x₁ + ε) | Verlust 1 → 2 = **1,000**, 2 → 1 = **1,1249**, Differenz **−0,1249** = ½ ρ² (Var₁ − Var₂) mit ρ² = 0,3902; standardisiert Var₁ = Var₂ und die Differenz ist **0**: ein Gleichstand. | `test_hand_values_of_the_text` |
| Zufalls-DAGs, 6 Knoten, n = 1 000, **20 DAGs** | Varsortability **0,96 roh**, **0,50 standardisiert**. Roh: NOTEARS **93,5 %** der wahren Kanten richtig gerichtet (2,4 % umgekehrt, SHD **0,40**), Rangbaseline **92,7 %** (6,5 %, SHD **0,75**), PC 39,8 % (27,6 % ungerichtet, SHD 2,90). Standardisiert: NOTEARS etwa **32 %** richtig und etwa **36 % umgekehrt** (SHD 4,6; je Plattform um eine Kante verschieden), Rangbaseline 45,5 % und 48,0 % (SHD 6,00), PC unverändert. | `test_experiment_1_random_dags` |
| Die Welt der Linie, n = 1 000, **16 Läufe** je Zelle | Roh: S → Z in **100 %**, Z → X in **100 %**, ganzer DAG in **100 %** richtig; PC lässt **40 %** der wahren Kanten ungerichtet. Standardisiert: **0 %**, 0 %, 0 %; nur **2,5 %** der Kanten richtig, **73,8 % umgekehrt**, im Mittel 1,19 Kanten fehlend. | `test_experiment_2_the_world` |
| Und mit n = 300? | Roh: S → Z **100 %**, Z → X **93,8 %**, ganzer DAG **87,5 %**. Standardisiert: **0 %** ganze DAGs, 5 % der Kanten richtig, 78,8 % umgekehrt: **mehr Standorte helfen nicht.** | `test_experiment_2_the_world` |
| Rauschstärke σ der Saison (n = 1 000, **20 Läufe**; Var(S) = σ², Var(Z) = 0,64 σ² + 1; die Ordnung kippt bei σ = 1,67) | S → Z richtig: σ = 1,0: NOTEARS **100 %**, Rangbaseline 100 %; σ = 1,5 (Var 2,25 gegen 2,45): NOTEARS **40 %**, Rangbaseline **100 %**; σ = 2,0 (Var 4,0 gegen 3,5): **0 %** und **0 %**. NOTEARS kippt vor der Varianzordnung. | `test_experiment_3_noise_of_the_root` |
| Die Nebenbedingung zum Anschauen (Voreinstellung) | **14 Teilprobleme**, ρ von 1 auf 10⁷, α bis 419,9; h(W) fällt von **0,112** auf **1,3·10⁻⁵**; der Verlust steigt von **1,796** auf **2,444** (die Nebenbedingung kostet Anpassung); Kanten über 0,3: 6 → 5. Ohne Nebenbedingung entsteht in **30 von 30** Läufen ein Kreis. | `test_experiment_4_constraint` |
| λ und Schwelle auf der Welt (**16 Läufe** je λ, roh, n = 1 000) | λ = 0,1: Schwelle 0,1 und 0,3 **SHD 0,00**, Schwelle 0,5 SHD 0,56; λ = 0,3: Schwelle 0,1 SHD 0,00, Schwelle 0,3 SHD **0,31**, Schwelle 0,5 SHD **2,69**; λ = 0,01: SHD 0,38 bis 0,50. Die Strafe schrumpft die Gewichte (wahr 0,6, geschätzt bei λ = 0,1 etwa 0,5), die feste Schwelle schneidet ab. | `test_experiment_4_constraint` |

Handwerte des Textes (`test_hand_values_of_the_text`): h für zwei Größen im Kreis mit Gewichten 1: 2 cosh 1 − 2 = **1,0862**; Dreier-Kreis: e + 2e^(−½) cos(√3/2) − 3 = **0,5042**; DAG: **0**. Varianzen der Welt: S **1**, Z **1,64**, X **2,0496**, Y **3,6329**, W **1**.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleiches Rauschen und unveränderte Skala** | Die Richtungen kommen aus der Varianzordnung: Eine Standardisierung oder ein Rauschen, das die Ordnung kippt, löscht sie oder dreht sie um (Experimente 1 bis 3). | Über die Äquivalenzklasse hinaus |
| **Linear und Gauß** | Das Modell ist eine lineare Gewichtsmatrix; nichtlineare Zusammenhänge sind nicht gebaut. | nicht gebaut |
| **Alles Wichtige ist gemessen** | Verborgene Ursachen sind nicht gebaut (Stück 13 zeigt, was sie tun). | Wann geht es gar nicht? |
| **Strafe und Schwelle** | Die L1-Strafe schrumpft die Gewichte, die feste Schwelle schneidet ab; beide zusammen wirken (Experiment 4). | nicht gebaut |
| **Lokale Lösung** | Die Aufgabe ist nicht konvex; der Löser findet einen Punkt, keinen Beweis des globalen Optimums. | nicht gebaut |
| **Kleine Graphen** | Fünf bis sechs Knoten; die Rechenzeit wächst mit d³, jede Anpassung dauert ein bis zwei Sekunden. | nicht gebaut |

Die Befunde, die **Konstruktion** sind, stehen als solche in `test_claims.py`: dass die Varsortability standardisierter Daten 0,5 ist (alle Varianzen gleich), dass ohne Nebenbedingung Zyklen entstehen (das Modell verbietet sie nicht, die Regression wählt beide Richtungen) und dass der Verlust zweier Richtungen exakt ½ ρ² (Var₁ − Var₂) ist (eine Rechenidentität; der Befund ist, was daraus für Standardisierung und Rauschstärke folgt).
Die Zahlen hängen an der kleinen Zahl der festen Wiederholungen (20, 16, 20): Raten sind auf etwa ±10 Prozentpunkte genau, die Richtung der Aussagen ist es, nicht die Dezimale. Dass NOTEARS roh knapp besser ist als die Rangbaseline (SHD 0,40 gegen 0,75), liegt an deren überzähligen Kanten (t-Schwelle 3); die Richtungen sind gleich gut (93,5 % gegen 92,7 %).
**Standardisierte Daten sind plattformabhängig:** Alle Varianzen sind 1, der Verlust beider Richtungen ist gleich, und Rundungsrauschen entscheidet, wohin NOTEARS kippt; die Zahlen können je Rechner und Bibliotheksversion um einzelne Kanten abweichen (Zufalls-DAGs: 39 gegen 40 von 123 richtig). Die Rangbaseline löst den Gleichstand der Varianzen mit einer festen Zufallsreihenfolge, nicht mit der Spaltennummer (die in simulierten Welten die kausale Reihenfolge verriete) und nicht mit Rundungsrauschen.
Auf standardisierten Daten richtet NOTEARS in der Welt der Linie 74 % der Kanten **umgekehrt**, in den Zufalls-DAGs 36 %: Das ist eine Aussage über diese Läufe, kein „schlechter als der Zufall“, denn nach der Standardisierung entscheidet nur noch der Rest der Rechnung. Der eigene Löser (FISTA, Toleranz 10⁻⁵) gibt auf der Welt und in mindestens neun von zehn Zufalls-DAGs dieselbe Kantenmenge wie die scipy-Fassung der Originalarbeit (`test_oracle_notears.py`).
NOTEARS, Varsortability (Reisach/Seiler/Weichwald 2021), die Rangbaseline und die Identifizierbarkeit bei gleichen Rauschvarianzen (Peters/Bühlmann 2014) sind aus dem Gedächtnis geschrieben und nicht gegen die Quellen geprüft; die Implementierung ist durch die Gegenproben belegt.

## Bezug zu den Nachbarn im Portfolio

- [strukturlernen-demo](https://github.com/sebastian-hanisch/strukturlernen-demo) (Stück 13): PC und die Äquivalenzklasse; Welt, `stl_pc.py` und `stl_world.py` kommen wortgleich mit, PC ist hier der Vergleich. Dort bleiben Kanten offen, hier richtet NOTEARS sie, aber aus der Skala.
- [lagrange-kkt-demo](https://github.com/sebastian-hanisch/lagrange-kkt-demo), [straf-barriere-demo](https://github.com/sebastian-hanisch/straf-barriere-demo), [quasi-newton-demo](https://github.com/sebastian-hanisch/quasi-newton-demo): Lagrange-Multiplikatoren, quadratische Strafe, L-BFGS. Das **augmentierte** Lagrange-Verfahren (Strafe plus Multiplikator) steht dort nicht; hier nur als Mittel für eine Nebenbedingung (Azyklizität). Der Löser ist ein proximaler Gradient, nicht L-BFGS.
- [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo): L1 und FISTA für lineare inverse Probleme; hier dieselbe Technik als Werkzeug für die L1-Strafe, nicht als Thema.
- [sensitivitaet-demo](https://github.com/sebastian-hanisch/sensitivitaet-demo) und [identifizierbarkeit-demo](https://github.com/sebastian-hanisch/identifizierbarkeit-demo): setzen das Diagramm voraus; ob es sich aus Daten lernen lässt, ist die Frage der Stücke 13 bis 15.

## Tests

Hand-Instanzen (h für Zwei- und Dreier-Kreis und DAG, Gradient gegen Differenzen, Verlust beider Richtungen 1,000 und 1,1249 und der Gleichstand, Varsortability 1,0 und 0,5, Rangbaseline auf einer Kette), `expm` gegen `scipy.linalg.expm`, der Löser gegen die scipy-Fassung der Originalarbeit (L-BFGS-B), der Verlust zweier Richtungen gegen zwei Regressionen, Auswertung, Verdikte, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Baukasten für h, Experimente, Permalink, PDF), Footer und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `kausal_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Normalverteilung, Strukturmodell) |
| `dsep_graph.py` | d-Separation und Äquivalenzklassen (wortgleich aus Stück 2) |
| `stl_world.py`, `stl_pc.py`, `stl_constants.py` | Welt, PC und Konstanten aus Stück 13 (wortgleich) |
| `nts_notears.py` | `expm`, h und Gradient, augmentiertes Lagrange-Verfahren mit proximalem Gradienten, Schwelle bis azyklisch |
| `nts_baselines.py` | Varsortability, Rangbaseline, Verlust zweier Richtungen, Standardisierung, Vergleichsmaße |
| `nts_evaluation.py` | Einstellungen, Analyse, vier Experimente, Baukasten für h, Verdikte |
| `nts_visualization.py`, `nts_pdf_export.py` | Figuren, Tabellen, PDF |
| `nts_constants.py`, `nts_presets.py` | Regler, Experimente, Presets, Permalink |
| `tests/` | Hand-Instanzen, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Nichtlineare Varianten** (NOTEARS-MLP, DAGMA, GOLEM) und andere Azyklizitätsmaße: nur im Grenzen-Absatz genannt.
- **Verborgene Größen, Vorwissen, höhere Dimension:** fünf bis sechs Knoten; Verborgenes zeigt Stück 13.
- **L-BFGS-B als Löser:** scipy ist nur Entwicklungspaket (Orakel im Test), zur Laufzeit rechnet numpy.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html).
