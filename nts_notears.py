"""NOTEARS (Zheng u. a. 2018) in numpy: lineares Strukturmodell als stetige Optimierung. Minimiere den Kleinste-Quadrate-Verlust plus L1-Strafe unter der Nebenbedingung
h(W) = tr(exp(W∘W)) − d = 0, die genau für azyklische Gewichtsmatrizen gilt; Lösung durch ein augmentiertes Lagrange-Verfahren, dessen Teilprobleme ein proximaler Gradient mit
Beschleunigung (FISTA) löst. W[i, j] ist das Gewicht der Kante i → j. Kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_TAYLOR = 19
MAX_RHO_STEPS = 20            # ρ von 1 auf 10⁸ braucht acht Schritte
MAX_LINE_SEARCH = 60        # 2⁻⁶⁰ ist unter jeder Schrittweite, die noch etwas ändert
MAX_STEPS = 30000           # Obergrenze aller Gradientenschritte eines Laufs (die Voreinstellung braucht etwa 1 700)


def expm(a: np.ndarray) -> np.ndarray:
    """Matrixexponential durch Skalieren und Quadrieren mit Taylorreihe."""
    norm = np.abs(a).sum(axis=1).max()
    s = max(0, int(math.ceil(math.log2(max(norm, 1e-12) / 0.5))))
    b = a / (2 ** s)
    term = np.eye(a.shape[0])
    out = term.copy()
    for k in range(1, MAX_TAYLOR + 1):
        term = term @ b / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def h_value(w: np.ndarray) -> float:
    """h(W) = tr(exp(W∘W)) − d: null genau für Gewichtsmatrizen ohne gerichteten Kreis, sonst positiv."""
    return float(np.trace(expm(w * w)) - w.shape[0])


def h_and_grad(w: np.ndarray):
    """h(W) und der Gradient exp(W∘W)ᵀ ∘ 2W."""
    e = expm(w * w)
    return float(np.trace(e) - w.shape[0]), e.T * 2.0 * w


def soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def is_acyclic(edges) -> bool:
    """Kahn: Gibt es eine Reihenfolge der Knoten, in der jede Kante vorwärts zeigt?"""
    edges = set(edges)
    nodes = {v for e in edges for v in e}
    indeg = {v: 0 for v in nodes}
    for _, b in edges:
        indeg[b] += 1
    stack = [v for v in nodes if indeg[v] == 0]
    seen = 0
    while stack:
        v = stack.pop()
        seen += 1
        for a, b in edges:
            if a == v:
                indeg[b] -= 1
                if indeg[b] == 0:
                    stack.append(b)
    return seen == len(nodes)


@dataclass
class NotearsResult:
    w: np.ndarray
    steps: int                      # innere Schritte des proximalen Gradienten insgesamt
    h: float                        # h(W) am Ende
    trace: list = field(default_factory=list)       # je Teilproblem: ρ, α, h, Verlust, Kanten über 0,3


def notears(x: np.ndarray, lam: float = 0.1, max_outer: int = 30, h_tol: float = 1e-5, rho_max: float = 1e8, max_inner: int = 3000, tol: float = 1e-5, constrained: bool = True, max_steps: int = MAX_STEPS) -> NotearsResult:
    """Augmentiertes Lagrange-Verfahren: ρ wächst, bis h(W) auf höchstens ein Viertel fällt, dann α += ρ h; Abbruch bei h ≤ h_tol oder ρ ≥ ρ_max.
    constrained=False löst nur die L1-Regression jeder Spalte auf alle anderen (ohne Nebenbedingung). Die Daten werden zentriert."""
    x = x - x.mean(axis=0)
    n, d = x.shape
    s = x.T @ x / n
    eye = np.eye(d)
    mask = 1.0 - eye
    w = np.zeros((d, d))
    rho, alpha = (1.0, 0.0) if constrained else (0.0, 0.0)
    steps = 0
    trace: list = []

    def smooth(w_, rho_, alpha_):
        r = eye - w_
        loss = 0.5 * float(np.trace(r.T @ s @ r))
        h, gh = h_and_grad(w_)
        g = -s @ r + (alpha_ + rho_ * h) * gh
        return loss + alpha_ * h + 0.5 * rho_ * h * h, g * mask

    def loss_of(w_):
        r = eye - w_
        return 0.5 * float(np.trace(r.T @ s @ r))

    h_cur = float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_outer if constrained else 1):
            for _try in range(MAX_RHO_STEPS):
                if constrained and rho >= rho_max:
                    break
                if steps >= max_steps:
                    break
                y, t_prev, step = w.copy(), 1.0, 1.0
                for _it in range(max_inner):
                    f_y, g_y = smooth(y, rho, alpha)
                    for _ls in range(MAX_LINE_SEARCH):
                        w_new = soft_threshold(y - step * g_y, step * lam) * mask
                        f_new, _ = smooth(w_new, rho, alpha)
                        diff = w_new - y
                        if f_new <= f_y + np.sum(g_y * diff) + np.sum(diff * diff) / (2 * step) + 1e-14:
                            break
                        step *= 0.5
                    t_new = (1 + math.sqrt(1 + 4 * t_prev * t_prev)) / 2
                    y = w_new + (t_prev - 1) / t_new * (w_new - w)
                    move = float(np.abs(w_new - w).max())
                    w, t_prev = w_new, t_new
                    steps += 1
                    step = min(step * 1.5, 1e3)
                    if move < tol or steps >= max_steps:
                        break
                h_new = h_value(w)
                trace.append({"rho": rho, "alpha": alpha, "h": h_new, "loss": loss_of(w), "edges": int((np.abs(w) > 0.3).sum())})
                if not constrained:
                    break
                if h_new > 0.25 * h_cur and rho < rho_max:
                    rho *= 10
                else:
                    break
            h_cur = h_value(w)
            alpha += rho * h_cur
            if h_cur <= h_tol or rho >= rho_max or not constrained or steps >= max_steps:
                break
    return NotearsResult(w, steps, h_value(w), trace)


def threshold_dag(w: np.ndarray, thr: float = 0.3) -> set:
    """Kanten (i, j) mit |W[i, j]| über der Schwelle; ist der Graph nicht azyklisch, wird die schwächste Kante gestrichen, bis er es ist (wie in der Originalarbeit)."""
    edges = {(i, j): abs(w[i, j]) for i in range(w.shape[0]) for j in range(w.shape[1]) if abs(w[i, j]) > thr}
    for _ in range(len(edges) + 1):
        if is_acyclic(edges):
            break
        del edges[min(edges, key=edges.get)]
    return set(edges)
