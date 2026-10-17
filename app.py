"""Kontinuierliches Strukturlernen: Was NOTEARS wirklich sortiert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vierzehntes Stück der Linie Kausale Diagramme und Adjustierung. NOTEARS findet ein Diagramm durch Optimieren (Kleinste-Quadrate-Verlust, L1-Strafe, Azyklizität als Gleichung h(W) = 0, augmentiertes Lagrange-Verfahren)
und richtet dabei Kanten, die PC offen lässt; die Richtungen stammen aus der Varianzordnung der Daten und verschwinden mit einer Standardisierung.
Kern: nts_notears.py, nts_baselines.py, nts_evaluation.py; PC und Welt: stl_pc.py, stl_world.py (wortgleich aus Stück 13).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import nts_constants as C
from nts_evaluation import (
    BUILDER_ARROWS,
    BUILDER_NODES,
    Settings,
    analyse,
    builder_result,
    constraint_experiment,
    de,
    random_experiment,
    sigma_experiment,
    verdict_builder,
    verdict_constraint,
    verdict_instance,
    verdict_pairs,
    verdict_random,
    verdict_sigma,
    verdict_world,
    world_experiment,
)
from nts_pdf_export import build_pdf
from nts_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from nts_visualization import (
    build_builder_diagram,
    build_builder_table,
    build_edge_table,
    build_grid_table,
    build_learned_diagram,
    build_methods_table,
    build_pair_chart,
    build_pair_table,
    build_random_chart,
    build_random_table,
    build_sigma_chart,
    build_sigma_table,
    build_summary_table,
    build_trace_chart,
    build_trace_table,
    build_true_diagram,
    build_world_chart,
    build_world_table,
    style_edge_table,
)

st.set_page_config(page_title="NOTEARS: Was wirklich sortiert wird – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme (NOTEARS braucht ein bis zwei Sekunden je Anpassung) …"
ARROW_KEYS = {a: f"hb_{BUILDER_NODES[a[0]]}{BUILDER_NODES[a[1]]}" for a in BUILDER_ARROWS}
BUILDER_EXAMPLES = {
    "Zwei-Zyklus A ⇄ B": [(0, 1), (1, 0)],
    "Drei-Zyklus A → B → C → A": [(0, 1), (1, 2), (2, 0)],
    "Kette A → B → C": [(0, 1), (1, 2)],
    "Dreieck ohne Kreis": [(0, 1), (1, 2), (0, 2)],
}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _random():
    return random_experiment()


@st.cache_data(show_spinner=SPINNER)
def _world():
    return world_experiment()


@st.cache_data(show_spinner=SPINNER)
def _sigma():
    return sigma_experiment()


@st.cache_data(show_spinner=SPINNER)
def _constraint():
    return constraint_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def set_builder_example(name):
    for arrow, key in ARROW_KEYS.items():
        st.session_state[key] = arrow in BUILDER_EXAMPLES[name]


def experiment_button(label, key):
    if st.button(label, key=f"{key}_run"):
        st.session_state[f"{key}_ready"] = True
    return st.session_state.get(f"{key}_ready", False)


st.title("🧮 Kontinuierliches Strukturlernen")
st.markdown(
    """
**Gibt Optimieren die Richtungen, die Testen offen lässt?** Dies ist das vierzehnte Stück der Linie **Kausale Diagramme und Adjustierung**. PC (Stück 13) lässt Kanten ungerichtet, weil Unabhängigkeiten sie nicht unterscheiden.
**NOTEARS** stellt die Frage anders: Welche Gewichtsmatrix **W** erklärt die Daten am besten (Kleinste Quadrate, dazu eine L1-Strafe), **unter der Gleichung h(W) = 0**, die genau für azyklische Graphen gilt? Gelöst wird das mit einem
augmentierten Lagrange-Verfahren. Das Ergebnis richtet auf Rohdaten alle Kanten, aber **die Richtungen stammen aus der Varianz**: Eine Standardisierung löscht sie, und eine Rangbaseline, die nur Varianzen sortiert, macht roh dasselbe.

Weiter unten: der **Verlust zweier Richtungen** in geschlossener Form, der **Lauf der Nebenbedingung**, ein **Baukasten für h(W)**, vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So liest man das Ergebnis", expanded=True):
    st.markdown(
        """
1. **Die Welt** (links) ist die der Linie: Saison S → Nachfrage Z → Faktor X → Servicegrad Y, dazu Z → Y und Wetter W → Y, überall Einheitsrauschen.
2. **NOTEARS** (rechts): **grün** richtig gerichtet, **orange** umgekehrt, **rot** überzählig, **grau gestrichelt** fehlend. Gewichte unter der Schwelle werden gestrichen.
3. **Rangbaseline:** die Variablen nach Varianz ordnen und jede auf ihre Vorgänger regressieren. Sie schätzt keine Struktur; wenn sie ähnlich gut ist, sortiert NOTEARS nur nach Varianz.
4. **Skala:** „standardisiert“ bringt jede Spalte auf Streuung 1. PC ändert sich dadurch nicht (Korrelationen sind skalenfrei), NOTEARS und die Rangbaseline schon.
5. **Verlust zweier Richtungen:** Für ein Paar ist Verlust(i → j) − Verlust(j → i) = ½ ρ² (Var i − Var j): Es gewinnt die Richtung, in der die Größe mit der kleineren Varianz die Ursache ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
names = list(C.PRESETS)
for chunk in (names[:4], names[4:]):
    cols = st.columns(4)
    for col, name in zip(cols, chunk):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    scale = st.radio("Skala der Daten", C.SCALES, key="scale_radio", help="Roh: die Daten, wie die Welt sie erzeugt. Standardisiert: jede Spalte auf Mittel 0 und Streuung 1 (die Einheit der Messung fällt weg).")
    sigma = st.slider("Rauschstärke der Saison σ", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider",
                      help="Streuung des Rauschens der Wurzel S (1 = Einheitsrauschen wie überall). Var(S) = σ², Var(Z) = 0,64 σ² + 1: Bei σ = 1,67 kippt die Varianzordnung.")
    n = st.slider("Standorte", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Zeilen der Datenmatrix.")
    st.markdown("**Verfahren**")
    lam = st.select_slider("L1-Strafe λ", options=C.LAMBDAS, key="lam_slider", help="Je größer, desto stärker schrumpfen die Gewichte (wahr 0,6, geschätzt bei λ = 0,1 etwa 0,5).")
    thr = st.slider("Schwelle auf |W|", *bounds("thr_slider"), step=C.THR_STEP, key="thr_slider", help="Gewichte darunter zählen nicht als Kante; bei einem Kreis wird die schwächste Kante gestrichen.")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"scale_radio": scale, "sigma_slider": round(float(sigma), 2), "n_slider": int(n), "lam_slider": lam, "thr_slider": round(float(thr), 2), "seed_input": int(seed)})

settings = Settings(scale, round(float(sigma), 2), int(n), float(lam), round(float(thr), 2), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
(c1, c2), (c3, c4) = st.columns(2), st.columns(2)
nt, rk, pc = a.compare["NOTEARS"], a.compare["Rangbaseline"], a.compare["PC"]
c1.metric("NOTEARS: richtig gerichtet", f"{nt['correct']} von {nt['n_true']}", delta=f"{nt['reversed']} umgekehrt, SHD {de(nt['shd'], 1)}", delta_color="off", help="Wahre Kanten, die NOTEARS mit richtiger Richtung findet.")
c2.metric("Rangbaseline: richtig gerichtet", f"{rk['correct']} von {rk['n_true']}", delta=f"{rk['reversed']} umgekehrt, SHD {de(rk['shd'], 1)}", delta_color="off", help="Nur Varianzen sortiert, dann Regression auf die Vorgänger.")
c3.metric("PC: richtig gerichtet", f"{pc['correct']} von {pc['n_true']}", delta=f"{pc['undirected']} ungerichtet, SHD {de(pc['shd'], 1)}", delta_color="off", help="PC (stabil, Fisher-z, α = 0,01) auf denselben Daten; ungerichtete Kanten zählen im SHD halb.")
c4.metric("Varsortability", de(a.varsort, 2), help="Anteil der Paare (Vorfahre, Nachkomme) mit größerer Varianz beim Nachkommen: 1 = Varianz wächst längs jedes Wegs, 0,5 = Zufall.")

left, right = st.columns(2)
with left:
    st.plotly_chart(build_true_diagram(tuple(tuple(sorted(p for p, c in a.truth if c == v)) for v in range(5))), width="stretch", key="true_chart")
    st.caption("Das wahre Diagramm der Welt. X (grün umrandet) ist der Faktor, Y (orange umrandet) der Servicegrad.")
with right:
    st.plotly_chart(build_learned_diagram(a), width="stretch", key="learned_chart")
    st.caption("Was NOTEARS aus den Daten baut, gefärbt nach dem Urteil gegen das wahre Diagramm.")
verdict(verdict_instance(a))
st.dataframe(style_edge_table(build_edge_table(a)), hide_index=True, width="stretch")
st.dataframe(build_methods_table(a), hide_index=True, width="stretch")
st.dataframe(build_summary_table(a), hide_index=True, width="stretch")

if st.button("📄 PDF der aktuellen Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="notears.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Der Verlust zweier Richtungen ---------------------------------------------------------------------------------------------------------

st.subheader("⚖️ Warum eine Richtung gewinnt: der Verlust beider Richtungen")
st.caption("Für jedes wahre Paar die beiden Kleinste-Quadrate-Verluste ½ (Var(Ursache) + Restvarianz der Wirkung), gerechnet aus der Kovarianz der Daten. Roh gewinnt die wahre Richtung; standardisiert sind die Varianzen gleich, und es gibt einen Gleichstand.")
st.plotly_chart(build_pair_chart(a), width="stretch", key="pair_chart")
verdict(verdict_pairs(a))
st.dataframe(build_pair_table(a), hide_index=True, width="stretch")

st.markdown("---")

# --- Die Nebenbedingung --------------------------------------------------------------------------------------------------------------------

st.subheader("🔒 Die Nebenbedingung h(W) = 0")
st.caption("Das augmentierte Lagrange-Verfahren löst eine Folge von Teilproblemen mit wachsender Strafe ρ und Multiplikator α, bis h(W) = tr(exp(W∘W)) − d praktisch null ist. Links der Lauf dieser Einstellung, darunter ein Baukasten für h von Hand.")
st.plotly_chart(build_trace_chart(a.trace), width="stretch", key="trace_chart")
st.dataframe(build_trace_table(a.trace), hide_index=True, width="stretch")

for arrow, key in ARROW_KEYS.items():
    st.session_state.setdefault(key, arrow in BUILDER_EXAMPLES["Drei-Zyklus A → B → C → A"])
ex_cols = st.columns(len(BUILDER_EXAMPLES))
for col, name in zip(ex_cols, BUILDER_EXAMPLES):
    with col:
        st.button(name, width="stretch", on_click=set_builder_example, args=(name,), key=f"hb_example_{name}")
arrow_cols = st.columns(len(BUILDER_ARROWS))
for col, arrow in zip(arrow_cols, BUILDER_ARROWS):
    with col:
        st.checkbox(f"{BUILDER_NODES[arrow[0]]} → {BUILDER_NODES[arrow[1]]}", key=ARROW_KEYS[arrow])
weight = st.slider("Gewicht aller gewählten Pfeile", 0.25, 2.0, 1.0, 0.25, key="hb_weight")
selected = [arrow for arrow, key in ARROW_KEYS.items() if st.session_state[key]]
b = builder_result(selected, float(weight))
bl, br = st.columns(2)
with bl:
    st.plotly_chart(build_builder_diagram(selected, b["acyclic"]), width="stretch", key="builder_chart")
with br:
    st.metric("h(W)", de(b["h"], 4), help="tr(exp(W∘W)) − 3: null genau dann, wenn kein gerichteter Kreis vorliegt.")
    verdict(verdict_builder(b))
    if selected:
        st.dataframe(build_builder_table(b), hide_index=True, width="stretch")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Zufalls-DAGs, roh und standardisiert")
st.caption(f"{C.EXP_RANDOM_DAGS} feste Zufalls-DAGs mit {C.EXP_RANDOM_NODES} Knoten, n = {C.EXP_RANDOM_N}, Gewichte ±U(0,5; 1,5), Einheitsrauschen. NOTEARS, Rangbaseline und PC auf denselben Daten; Varsortability beider Fassungen.")
if experiment_button("▶ Experiment 1 rechnen (etwa 1 Minute)", "exp1"):
    res1 = _random()
    st.plotly_chart(build_random_chart(res1), width="stretch", key="random_chart")
    verdict(verdict_random(res1))
    st.dataframe(build_random_table(res1), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Die Welt der Linie, roh und standardisiert")
st.caption(f"{C.EXP_WORLD_REPS} feste Läufe je Zelle auf der Welt aus Stück 13, n = {' und '.join(str(n_) for n_ in C.EXP_WORLD_NS)}. Wird S → Z → X richtig gerichtet, was PC offen lässt?")
if experiment_button("▶ Experiment 2 rechnen (etwa 1 Minute)", "exp2"):
    res2 = _world()
    st.plotly_chart(build_world_chart(res2), width="stretch", key="world_chart")
    verdict(verdict_world(res2))
    st.dataframe(build_world_table(res2), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 3: Wenn die Varianzordnung kippt")
st.caption(f"Die Saison bekommt ein Rauschen der Stärke σ; {C.EXP_SIGMA_REPS} feste Läufe je Zelle, n = {C.EXP_SIGMA_N}. Anteil der Läufe mit S → Z richtig, NOTEARS gegen Rangbaseline.")
if experiment_button("▶ Experiment 3 rechnen (etwa 1 Minute)", "exp3"):
    res3 = _sigma()
    st.plotly_chart(build_sigma_chart(res3), width="stretch", key="sigma_chart")
    verdict(verdict_sigma(res3))
    st.dataframe(build_sigma_table(res3), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Die Nebenbedingung und die festen Zahlen")
st.caption(f"Ohne Nebenbedingung ({C.EXP_UNCONSTRAINED_REPS} Läufe) und mit ihr; dazu λ und die Schwelle gegeneinander ({C.EXP_GRID_REPS} feste Läufe je λ, n = {C.EXP_GRID_N}).")
if experiment_button("▶ Experiment 4 rechnen (etwa 1 Minute)", "exp4"):
    res4 = _constraint()
    verdict(verdict_constraint(res4))
    st.dataframe(build_grid_table(res4), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gleiches Rauschen und unveränderte Skala** | Die Richtungen kommen aus der Varianzordnung: Eine Standardisierung oder ein Rauschen, das die Ordnung kippt, löscht sie oder dreht sie um (Experimente 1 bis 3). | Über die Äquivalenzklasse hinaus |
| **Linear und Gauß** | Das Modell ist eine lineare Gewichtsmatrix; nichtlineare Zusammenhänge sind nicht gebaut. | nicht gebaut |
| **Alles Wichtige ist gemessen** | Verborgene Ursachen sind nicht gebaut (Stück 13 zeigt, was sie tun). | Wann geht es gar nicht? |
| **Strafe und Schwelle** | Die L1-Strafe schrumpft die Gewichte, die feste Schwelle schneidet ab; beide zusammen wirken (Experiment 4). | nicht gebaut |
| **Lokale Lösung** | Die Aufgabe ist nicht konvex; der Löser findet einen Punkt, keinen Beweis des globalen Optimums. | nicht gebaut |
| **Kleine Graphen** | Fünf bis sechs Knoten; die Rechenzeit wächst mit d³, jede Anpassung dauert ein bis zwei Sekunden. | nicht gebaut |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Lineares Strukturmodell $X = XW + \varepsilon$ mit der Gewichtsmatrix $W$ ($W_{ij}$: Kante $i \to j$, Diagonale null). Gesucht wird $W$ mit
$$\min_W \; \tfrac{1}{2n}\,\lVert X - XW \rVert_F^2 + \lambda \lVert W \rVert_1 \quad \text{unter} \quad h(W) = \operatorname{tr}\!\big(e^{W\circ W}\big) - d = 0.$$
Es gilt $h(W) \ge 0$ und $h(W) = 0$ genau dann, wenn der Graph von $W$ keinen gerichteten Kreis hat; der Gradient ist $\nabla h(W) = \big(e^{W\circ W}\big)^{\top} \circ 2W$. Beispiele: Zwei-Zyklus mit Gewichten 1: $h = 2\cosh 1 - 2 = 1{,}0862$;
Drei-Zyklus: $e + 2e^{-1/2}\cos(\sqrt{3}/2) - 3 = 0{,}5042$.

**Augmentiertes Lagrange-Verfahren.** Löse für wachsendes $\rho$ und Multiplikator $\alpha$ das Teilproblem $\min_W \; \text{Verlust}(W) + \lambda\lVert W\rVert_1 + \alpha\, h(W) + \tfrac{\rho}{2} h(W)^2$ (hier mit einem proximalen Gradienten mit Beschleunigung;
die L1-Strafe wirkt über die Softthreshold); erhöhe $\rho$ um den Faktor 10, bis $h$ auf höchstens ein Viertel fällt, setze $\alpha \leftarrow \alpha + \rho\, h(W)$; Abbruch bei $h \le 10^{-5}$ oder $\rho \ge 10^{8}$.

**Zwei Größen.** Mit zentrierten Daten und Kovarianz $\begin{pmatrix} v_1 & c \\ c & v_2 \end{pmatrix}$, $\rho^2 = c^2/(v_1 v_2)$ ist der Verlust der Richtung $1 \to 2$ gleich $\tfrac12\,(v_1 + v_2 - c^2/v_1)$, der von $2 \to 1$ gleich $\tfrac12\,(v_1 + v_2 - c^2/v_2)$, also
$$L(1 \to 2) - L(2 \to 1) = \tfrac12\,\rho^2\,(v_1 - v_2).$$
Es gewinnt die Richtung, in der die Größe mit der kleineren Varianz die Ursache ist; bei $v_1 = v_2$ (Standardisierung) gibt es einen Gleichstand.

**Varsortability.** Anteil der Paare (Vorfahre $i$, Nachkomme $j$) mit $\operatorname{Var}(X_j) > \operatorname{Var}(X_i)$; ein Gleichstand zählt halb. **Rangbaseline:** Variablen nach wachsender Varianz ordnen, jede auf alle Vorgänger regressieren, Kanten mit $|t| < 3$ streichen.

**Literatur.** X. Zheng, B. Aragam, P. Ravikumar, E. Xing, *DAGs with NO TEARS: Continuous Optimization for Structure Learning*, NeurIPS 2018; A. Reisach, C. Seiler, S. Weichwald, *Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game*, NeurIPS 2021;
J. Peters, P. Bühlmann, *Identifiability of Gaussian structural equation models with equal error variances*, Biometrika 2014 (alle aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kausale Diagramme: von Konfundierung bis Wahl](https://sebastianhanisch.net/konzepte-kausale-diagramme.html)."
)
