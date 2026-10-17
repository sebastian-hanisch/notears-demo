"""Gegenproben ohne Strukturschätzung und Vergleichsmaße: Varsortability, die Rangbaseline (Variablen nach Varianz ordnen, auf Vorgänger regressieren), der Verlust zweier Richtungen
in geschlossener Form, Standardisierung und der Vergleich eines Graphen mit dem wahren Diagramm. Kein Streamlit."""

from __future__ import annotations

import numpy as np

from kausal_generator import SplitMix64

TIE = 1e-9


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)


def reach_matrix(edges, d: int) -> np.ndarray:
    """reach[i, j]: Es gibt einen gerichteten Weg von i nach j (mindestens eine Kante)."""
    adj = np.zeros((d, d), dtype=bool)
    for i, j in edges:
        adj[i, j] = True
    reach = adj.copy()
    for _ in range(d):
        reach = reach | ((reach.astype(int) @ adj.astype(int)) > 0)
    return reach


def varsortability(edges, x: np.ndarray) -> float:
    """Anteil der Paare (Vorfahre i, Nachkomme j) mit größerer Varianz bei j (Reisach u. a.): 1 = die Varianz wächst längs jedes Wegs, 0,5 = nach dem Zufall. Ein Gleichstand (Toleranz 1e-9) zählt halb."""
    d = x.shape[1]
    reach = reach_matrix(edges, d)
    var = x.var(axis=0)
    total = good = 0.0
    for i in range(d):
        for j in range(d):
            if reach[i, j]:
                total += 1
                if abs(var[j] - var[i]) <= TIE * max(var[i], var[j]):
                    good += 0.5
                elif var[j] > var[i]:
                    good += 1.0
    return good / total if total else float("nan")


def variance_order(x: np.ndarray, tie_seed: int = 0) -> list:
    """Spalten nach wachsender Varianz (auf 8 Stellen gerundet). Bei Gleichstand, etwa nach der Standardisierung, entscheidet eine feste Zufallsreihenfolge (aus tie_seed), weder die Spaltennummer
    (die in simulierten Welten die kausale Reihenfolge verraten könnte) noch Rundungsrauschen (das je Plattform verschieden ausfiele)."""
    var = np.round(x.var(axis=0), 8)
    tie = SplitMix64(tie_seed).uniform(x.shape[1])
    return [int(v) for v in sorted(range(x.shape[1]), key=lambda j: (var[j], tie[j]))]


def sortnregress(x: np.ndarray, t_min: float = 3.0, tie_seed: int = 0) -> set:
    """Rangbaseline: Variablen nach wachsender Varianz ordnen, jede auf alle Vorgänger regressieren, Kanten mit |t| unter t_min streichen. Ergebnis: Kantenmenge {(i, j)} für i → j."""
    x = x - x.mean(axis=0)
    n, d = x.shape
    order = variance_order(x, tie_seed)
    edges = set()
    for pos in range(1, d):
        j = order[pos]
        preds = order[:pos]
        z = np.column_stack([x[:, p] for p in preds])
        beta, *_ = np.linalg.lstsq(z, x[:, j], rcond=None)
        res = x[:, j] - z @ beta
        cov = np.linalg.inv(z.T @ z) * (res @ res) / (n - z.shape[1])
        t = beta / np.sqrt(np.diag(cov))
        edges |= {(p, j) for k, p in enumerate(preds) if abs(t[k]) >= t_min}
    return edges


def pair_losses(cov: np.ndarray, i: int, j: int) -> dict:
    """Kleinste-Quadrate-Verlust ½ (Var(Ursache) + Restvarianz der Wirkung) der beiden Richtungen eines Paars aus der Kovarianzmatrix (zentrierte Daten).
    Verlust(i → j) − Verlust(j → i) = ½ ρ² (Var_i − Var_j)."""
    vi, vj, cij = cov[i, i], cov[j, j], cov[i, j]
    rho2 = cij * cij / (vi * vj)
    loss_ij = 0.5 * (vi + vj - cij * cij / vi)
    loss_ji = 0.5 * (vj + vi - cij * cij / vj)
    return {"loss_forward": loss_ij, "loss_backward": loss_ji, "difference": loss_ij - loss_ji, "rho2": rho2, "var_i": vi, "var_j": vj}


def compare(est_dir: set, est_undir: set, truth: set) -> dict:
    """Vergleich mit dem wahren Diagramm (Menge (i, j) für i → j): richtig gerichtet, umgekehrt, ungerichtet gefunden, fehlend, überzählig; SHD zählt Umkehrung 1 und ungerichtet 0,5."""
    skel_t = {frozenset(e) for e in truth}
    correct = len(est_dir & truth)
    reversed_ = len({(b, a) for a, b in est_dir} & truth)
    undirected = len({frozenset(e) for e in est_undir} & skel_t)
    found = {frozenset(e) for e in est_dir} | {frozenset(e) for e in est_undir}
    missing = len(skel_t - found)
    extra = len(found - skel_t)
    return {"correct": correct, "reversed": reversed_, "undirected": undirected, "missing": missing, "extra": extra, "n_true": len(truth), "shd": missing + extra + reversed_ + 0.5 * undirected}
