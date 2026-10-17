"""PDF-Export der aktuellen Auswertung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import stl_world as W
from nts_evaluation import METHODS, Analysis, de, verdict_instance

_REPLACE = {"→": "->", "←": "<-", "–": "-", "—": "-", "−": "-", "„": '"', "“": '"', "×": "x", "·": "*", "**": "", "σ": "sigma", "λ": "lambda", "ρ": "rho", "½": "1/2", "²": "2", "≤": "<=", "≥": ">=", "∘": "o"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kontinuierliches Strukturlernen: Was NOTEARS sortiert"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Welt der Linie (S, Z, X, Y, W), Daten {s.scale}, Rauschstärke der Wurzel S sigma = {de(s.sigma, 2)}, {s.n} Standorte, lambda = {de(s.lam, 2)}, Schwelle {de(s.thr, 2)}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    for x, y, status in a.status:
        pdf.multi_cell(0, 4.6, _s(f"NOTEARS-Kante {W.NAMES[x]} -> {W.NAMES[y]}: {status}."), new_x="LMARGIN", new_y="NEXT")
    for m in METHODS:
        c = a.compare[m]
        pdf.multi_cell(0, 4.6, _s(f"{m}: {c['correct']} von {c['n_true']} Kanten richtig gerichtet, {c['reversed']} umgekehrt, {c['undirected']} ungerichtet, {c['missing']} fehlend, {c['extra']} überzählig (SHD {de(c['shd'], 1)})."),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 4.6, _s(f"Varsortability der Daten: {de(a.varsort, 2)}; h(W) am Ende: {a.h:.2e}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5, _s(verdict_instance(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: notears-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
