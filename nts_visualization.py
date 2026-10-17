"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import nts_constants as C
import nts_evaluation as ev
import stl_world as W
from nts_evaluation import METHODS, Analysis, de, pct, signed

POS = {W.S: (0.0, 1.0), W.Z: (0.34, 1.0), W.X: (0.34, 0.0), W.Y: (0.80, 0.5), W.W: (1.0, 1.0)}
STATUS_COLORS = {"richtig": C.COLOR_GOOD, "umgekehrt": C.COLOR_TRUTH, "überzählig": C.COLOR_BAD, "fehlt": C.COLOR_NEUTRAL}
STATUS_LABELS = {"richtig": "richtig", "umgekehrt": "umgekehrt", "überzählig": "überzählig", "fehlt": "fehlt"}


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _segment(p0, p1, r: float = 0.11):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / ln, dy / ln
    return (p0[0] + ux * r, p0[1] + uy * r), (p1[0] - ux * r, p1[1] - uy * r)


def _arrow(fig: go.Figure, pos: dict, a: int, b: int, color: str, width: float = 3):
    s, e = _segment(pos[a], pos[b])
    fig.add_annotation(x=e[0], y=e[1], ax=s[0], ay=s[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=width, arrowcolor=color, text="")


def _nodes(fig: go.Figure, pos: dict, names, outlines=None):
    outlines = outlines or {}
    for v, (x, y) in pos.items():
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[names[v]], textposition="middle center", textfont=dict(size=12),
                                 marker=dict(size=64, color="rgba(47,107,101,0.12)", line=dict(width=2, color=outlines.get(v, "#1b1b1b"))), hoverinfo="skip", showlegend=False))


def _frame(fig: go.Figure, height: int = 330, **layout) -> go.Figure:
    fig.update_xaxes(visible=False, range=[-0.2, 1.2])
    fig.update_yaxes(visible=False, range=[-0.3, 1.3])
    return _fixed(fig, height, **layout)


def build_true_diagram(parents: tuple) -> go.Figure:
    fig = go.Figure()
    for c, ps in enumerate(parents):
        for p in ps:
            _arrow(fig, POS, p, c, "#6b6b6b")
    _nodes(fig, POS, W.NAMES, {W.X: C.COLOR_GOOD, W.Y: C.COLOR_TRUTH})
    return _frame(fig, showlegend=False)


def build_learned_diagram(a: Analysis) -> go.Figure:
    """Der von NOTEARS gelernte Graph nach Urteil gefärbt (richtig, umgekehrt, überzählig); fehlende wahre Kanten gestrichelt grau."""
    fig = go.Figure()
    present = set()
    for x, y, st in a.status:
        if st == "fehlt":
            s, e = _segment(POS[x], POS[y], 0.1)
            fig.add_trace(go.Scatter(x=[s[0], e[0]], y=[s[1], e[1]], mode="lines", line=dict(color=C.COLOR_NEUTRAL, width=2, dash="dash"), hoverinfo="skip", showlegend=False))
        else:
            _arrow(fig, POS, x, y, STATUS_COLORS[st], 4 if st != "richtig" else 3)
        present.add(st)
    _nodes(fig, POS, W.NAMES, {W.X: C.COLOR_GOOD, W.Y: C.COLOR_TRUTH})
    for st in ("richtig", "umgekehrt", "überzählig", "fehlt"):
        if st in present:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=STATUS_COLORS[st], width=4, dash="dash" if st == "fehlt" else "solid"), name=STATUS_LABELS[st]))
    return _frame(fig, showlegend=True)


def build_edge_table(a: Analysis) -> pd.DataFrame:
    rows = []
    for x, y, st in a.status:
        rows.append({"Kante": f"{W.NAMES[x]} → {W.NAMES[y]}", "Urteil": STATUS_LABELS[st], "Gewicht W (vor der Schwelle)": signed(a.w[x, y], 2), "Gegenrichtung": signed(a.w[y, x], 2)})
    return pd.DataFrame(rows)


def style_edge_table(df: pd.DataFrame):
    def row_style(row):
        return ["background-color: rgba(192,57,43,0.15)" if row["Urteil"] != "richtig" else ""] * len(row)

    return df.style.apply(row_style, axis=1)


def build_methods_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Verfahren": m, "richtig gerichtet": f"{a.compare[m]['correct']} von {a.compare[m]['n_true']}", "umgekehrt": a.compare[m]["reversed"], "ungerichtet": a.compare[m]["undirected"],
                          "fehlend": a.compare[m]["missing"], "überzählig": a.compare[m]["extra"], "SHD": de(a.compare[m]["shd"], 1)} for m in METHODS])


def build_summary_table(a: Analysis) -> pd.DataFrame:
    s = a.settings
    return pd.DataFrame([
        {"Größe": "Varsortability der Daten (1 = Varianz wächst längs jedes Wegs, 0,5 = Zufall)", "Wert": de(a.varsort, 2)},
        {"Größe": "Varianzen S, Z, X, Y, W", "Wert": ", ".join(de(v, 2) for v in a.variances)},
        {"Größe": "h(W) am Ende des Verfahrens", "Wert": f"{a.h:.1e}".replace(".", ",")},
        {"Größe": "Teilprobleme des augmentierten Lagrange-Verfahrens, Schritte des Gradienten", "Wert": f"{len(a.trace)}, {a.steps}"},
        {"Größe": "Standorte, λ, Schwelle", "Wert": f"{s.n}, {de(s.lam, 2)}, {de(s.thr, 2)}"},
    ])


def build_pair_chart(a: Analysis) -> go.Figure:
    labels = [f"{W.NAMES[r['a']]} / {W.NAMES[r['b']]}" for r in a.pairs]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["loss_forward"] for r in a.pairs], name="Verlust Ursache → Wirkung (wahre Richtung)", marker_color=C.COLOR_GOOD))
    fig.add_trace(go.Bar(x=labels, y=[r["loss_backward"] for r in a.pairs], name="Verlust Wirkung → Ursache (umgekehrt)", marker_color=C.COLOR_TRUTH))
    lo = min(min(r["loss_forward"], r["loss_backward"]) for r in a.pairs)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Kleinste-Quadrate-Verlust des Paars", range=[0.9 * lo, None])
    fig.update_xaxes(title="wahres Paar (Ursache / Wirkung)")
    return _fixed(fig, 340)


def build_pair_table(a: Analysis) -> pd.DataFrame:
    return pd.DataFrame([{"Paar": f"{W.NAMES[r['a']]} → {W.NAMES[r['b']]}", "Var(Ursache)": de(r["var_i"], 3), "Var(Wirkung)": de(r["var_j"], 3), "ρ²": de(r["rho2"], 3),
                          "Verlust wahre Richtung": de(r["loss_forward"], 4), "Verlust umgekehrt": de(r["loss_backward"], 4), "Differenz": signed(r["difference"], 4),
                          "½ ρ² (Var Ursache − Var Wirkung)": signed(0.5 * r["rho2"] * (r["var_i"] - r["var_j"]), 4), "bevorzugt": r["prefers"]} for r in a.pairs])


def build_trace_chart(trace: list) -> go.Figure:
    fig = go.Figure()
    idx = list(range(1, len(trace) + 1))
    fig.add_trace(go.Scatter(x=idx, y=[max(t["h"], 1e-12) for t in trace], mode="lines+markers", name="h(W)", line=dict(color=C.COLOR_BAD, width=3)))
    fig.add_trace(go.Scatter(x=idx, y=[t["rho"] for t in trace], mode="lines+markers", name="ρ", line=dict(color=C.COLOR_CONTROL, width=2, dash="dash")))
    fig.update_yaxes(type="log", title="h(W) und ρ (logarithmisch)")
    fig.update_xaxes(title="Teilproblem des augmentierten Lagrange-Verfahrens")
    return _fixed(fig, 340)


def build_trace_table(trace: list) -> pd.DataFrame:
    return pd.DataFrame([{"Schritt": k + 1, "ρ": f"{t['rho']:.0e}".replace(".", ","), "α": de(t["alpha"], 2), "h(W)": f"{t['h']:.1e}".replace(".", ","), "Verlust": de(t["loss"], 3), "Kanten über 0,3": t["edges"]} for k, t in enumerate(trace)])


# ----------------------------------------------------------------------------- Experimente

def build_random_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    labels, good, bad = [], [], []
    for sc in C.SCALES:
        for m in ("NOTEARS", "Rangbaseline", "PC"):
            r = res["rows"][sc][m]
            labels.append(f"{m}, {sc}")
            good.append(100 * r["correct"])
            bad.append(100 * r["reversed"])
    fig.add_trace(go.Bar(x=labels, y=good, name="richtig gerichtet", marker_color=C.COLOR_GOOD))
    fig.add_trace(go.Bar(x=labels, y=bad, name="umgekehrt", marker_color=C.COLOR_TRUTH))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="wahre Kanten (%)", range=[0, 104])
    fig.update_xaxes(tickangle=-30)
    return _fixed(fig, 420)


def build_random_table(res: dict) -> pd.DataFrame:
    rows = []
    for sc in C.SCALES:
        for m in METHODS:
            r = res["rows"][sc][m]
            rows.append({"Daten": sc, "Verfahren": m, "Varsortability": de(res["varsort"][sc], 2), "richtig gerichtet": pct(r["correct"], 1), "umgekehrt": pct(r["reversed"], 1), "ungerichtet": pct(r["undirected"], 1),
                         "fehlend je DAG": de(r["missing"], 2), "überzählig je DAG": de(r["extra"], 2), "SHD": de(r["shd"], 2)})
    return pd.DataFrame(rows)


def build_world_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    for sc, color in zip(C.SCALES, (C.COLOR_GOOD, C.COLOR_BAD)):
        rows = [r for r in res["rows"] if r["scale"] == sc]
        fig.add_trace(go.Bar(x=[f"n = {r['n']}" for r in rows], y=[100 * r["full"] for r in rows], name=f"NOTEARS, {sc}: ganzer DAG richtig", marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Läufe mit genau richtigem Diagramm (%)", range=[0, 104])
    return _fixed(fig, 340)


def build_world_table(res: dict) -> pd.DataFrame:
    rows = []
    for r in res["rows"]:
        m = r["methods"]
        rows.append({"n": r["n"], "Daten": r["scale"], "S → Z richtig": pct(r["sz"]), "Z → X richtig": pct(r["zx"]), "ganzer DAG richtig": pct(r["full"]), "NOTEARS: richtig / umgekehrt": f"{pct(m['NOTEARS']['correct'])} / {pct(m['NOTEARS']['reversed'])}",
                     "Rangbaseline: richtig / umgekehrt": f"{pct(m['Rangbaseline']['correct'])} / {pct(m['Rangbaseline']['reversed'])}", "PC: richtig / ungerichtet": f"{pct(m['PC']['correct'])} / {pct(m['PC']['undirected'])}",
                     "SHD NOTEARS / Rang / PC": f"{de(m['NOTEARS']['shd'], 2)} / {de(m['Rangbaseline']['shd'], 2)} / {de(m['PC']['shd'], 2)}"})
    return pd.DataFrame(rows)


def build_sigma_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["sigma"] for r in rows], y=[100 * r["notears"] for r in rows], mode="lines+markers", name="NOTEARS", line=dict(color=C.COLOR_GOOD, width=3)))
    fig.add_trace(go.Scatter(x=[r["sigma"] for r in rows], y=[100 * r["rank"] for r in rows], mode="lines+markers", name="Rangbaseline", line=dict(color=C.COLOR_CONTROL, width=3, dash="dash")))
    fig.add_vline(x=res["flip"], line=dict(color=C.COLOR_TRUTH, width=2), annotation_text=f"Varianzordnung kippt bei σ = {de(res['flip'], 2)}", annotation_position="top")
    fig.update_xaxes(title="Rauschstärke σ der Saison")
    fig.update_yaxes(title="Läufe mit S → Z richtig (%)", range=[-2, 104])
    return _fixed(fig, 340)


def build_sigma_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"σ": de(r["sigma"], 2), "Var(S) gemessen / exakt": f"{de(r['var_s'], 2)} / {de(r['var_s_exact'], 2)}", "Var(Z) gemessen / exakt": f"{de(r['var_z'], 2)} / {de(r['var_z_exact'], 2)}",
                          "NOTEARS: S → Z richtig": pct(r["notears"]), "Rangbaseline: S → Z richtig": pct(r["rank"])} for r in res["rows"]])


def build_grid_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"λ": de(r["lam"], 2), "Schwelle": de(r["thr"], 1), "SHD": de(r["shd"], 2), "DAG genau": pct(r["exact"]), "fehlend": de(r["missing"], 2), "überzählig": de(r["extra"], 2)} for r in res["grid"]])


# ----------------------------------------------------------------------------- Baukasten

BUILDER_POS = {0: (0.0, 0.0), 1: (0.5, 1.0), 2: (1.0, 0.0)}


def build_builder_diagram(selected, acyclic: bool) -> go.Figure:
    fig = go.Figure()
    for i, j in sorted(selected):
        _arrow(fig, BUILDER_POS, i, j, C.COLOR_GOOD if acyclic else C.COLOR_BAD, 3)
    _nodes(fig, BUILDER_POS, ev.BUILDER_NODES)
    fig.update_xaxes(visible=False, range=[-0.35, 1.35])
    fig.update_yaxes(visible=False, range=[-0.35, 1.35])
    return _fixed(fig, 280, showlegend=False)


def build_builder_table(b: dict) -> pd.DataFrame:
    w = b["w"]
    return pd.DataFrame([{"von": ev.BUILDER_NODES[i], "nach": ev.BUILDER_NODES[j], "Gewicht": de(w[i, j], 2)} for i in range(3) for j in range(3) if w[i, j] != 0])
