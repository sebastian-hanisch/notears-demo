"""Auswertung: eine Analyse der Welt mit NOTEARS, Rangbaseline und PC, die vier Experimente über feste Ströme, der Baukasten für h(W) und die Verdikte. Nur numpy und die Module der Linie, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import nts_baselines as B
import nts_constants as C
import nts_notears as N
import stl_pc as P
import stl_world as W
from kausal_generator import SplitMix64


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    x = 0.0 if abs(x) < 0.5 * 10 ** -digits * 1e-3 else x
    return ("+" if x >= 0 else "−") + de(abs(x), digits)


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits) + " %"


def _n(n: int) -> str:
    return f"{n:,}".replace(",", " ")


METHODS = ("NOTEARS", "Rangbaseline", "PC")


@dataclass(frozen=True)
class Settings:
    scale: str = C.DEFAULT_SCALE
    sigma: float = C.DEFAULT_SIGMA
    n: int = C.DEFAULT_N
    lam: float = C.DEFAULT_LAMBDA
    thr: float = C.DEFAULT_THR
    seed: int = C.DEFAULT_SEED

    @property
    def standardized(self) -> bool:
        return self.scale == C.SCALES[1]


def simulate_scaled(world: W.World, n: int, rng: SplitMix64, sigma: float = 1.0, standardized: bool = False) -> np.ndarray:
    """Daten der Welt (Spalten S, Z, X, Y, W) mit Rauschstärke σ der Wurzel S; auf Wunsch jede Spalte auf Mittel 0 und Streuung 1 gebracht."""
    e = rng.normal(5 * n).reshape(n, 5)
    s = sigma * e[:, W.S]
    z = world.a_sz * s + e[:, W.Z]
    x = world.a_zx * z + e[:, W.X]
    w = e[:, W.W]
    y = world.b_xy * x + world.b_zy * z + world.b_wy * w + e[:, W.Y]
    data = np.column_stack([s, z, x, y, w])
    return B.standardize(data) if standardized else data


def truth_edges(world: W.World) -> set:
    return {(p, c) for c, ps in enumerate(world.parents) for p in ps}


def edge_status(learned: set, truth: set) -> list:
    out = []
    for a, b in sorted(learned):
        out.append((a, b, "richtig" if (a, b) in truth else "umgekehrt" if (b, a) in truth else "überzählig"))
    skel = {frozenset(e) for e in learned}
    for a, b in sorted(truth):
        if frozenset((a, b)) not in skel:
            out.append((a, b, "fehlt"))
    return out


@dataclass
class Analysis:
    settings: Settings
    truth: set
    variances: list
    varsort: float
    w: np.ndarray               # NOTEARS-Gewichte vor der Schwelle
    learned: set                # NOTEARS-Kanten nach der Schwelle
    status: list                # (a, b, Urteil)
    compare: dict               # je Verfahren der Vergleich mit dem wahren Diagramm
    rank: set
    pc: tuple
    h: float
    trace: list
    pairs: list                 # je wahrer Kante die Verluste beider Richtungen
    steps: int


def analyse(s: Settings) -> Analysis:
    world = W.World()
    data = simulate_scaled(world, s.n, SplitMix64(s.seed), s.sigma, s.standardized)
    truth = truth_edges(world)
    res = N.notears(data, lam=s.lam)
    learned = N.threshold_dag(res.w, s.thr)
    rank = B.sortnregress(data, tie_seed=s.seed)
    corr = np.corrcoef(data, rowvar=False)
    pc = P.pc(5, P.fisher_z_test(corr, len(data), C.PC_ALPHA)).graph
    cov = np.cov(data, rowvar=False)
    pairs = []
    for a, b in sorted(truth):
        r = B.pair_losses(cov, a, b)
        tol = 1e-9 * max(r["loss_forward"], r["loss_backward"])
        r.update({"a": a, "b": b, "prefers": "a → b" if r["difference"] < -tol else "b → a" if r["difference"] > tol else "Gleichstand"})
        pairs.append(r)
    compare = {"NOTEARS": B.compare(learned, set(), truth), "Rangbaseline": B.compare(rank, set(), truth), "PC": B.compare(set(pc[0]), set(pc[1]), truth)}
    return Analysis(s, truth, [float(v) for v in data.var(axis=0)], B.varsortability(truth, data), res.w, learned, edge_status(learned, truth), compare, rank, pc, res.h, res.trace, pairs, res.steps)


# ----------------------------------------------------------------------------- Experiment 1: Zufalls-DAGs roh und standardisiert

def _aggregate(rows: list) -> dict:
    n_true = sum(r["n_true"] for r in rows)
    return {"correct": sum(r["correct"] for r in rows) / n_true, "reversed": sum(r["reversed"] for r in rows) / n_true, "undirected": sum(r["undirected"] for r in rows) / n_true,
            "missing": float(np.mean([r["missing"] for r in rows])), "extra": float(np.mean([r["extra"] for r in rows])), "shd": float(np.mean([r["shd"] for r in rows]))}


def _methods(data: np.ndarray, truth: set, tie_seed: int, lam: float = C.DEFAULT_LAMBDA, thr: float = C.DEFAULT_THR) -> dict:
    d = data.shape[1]
    out = {"NOTEARS": B.compare(N.threshold_dag(N.notears(data, lam=lam).w, thr), set(), truth), "Rangbaseline": B.compare(B.sortnregress(data, tie_seed=tie_seed), set(), truth)}
    g = P.pc(d, P.fisher_z_test(np.corrcoef(data, rowvar=False), len(data), C.PC_ALPHA)).graph
    out["PC"] = B.compare(set(g[0]), set(g[1]), truth)
    return out


def random_experiment(dags: int = C.EXP_RANDOM_DAGS, d: int = C.EXP_RANDOM_NODES, p: float = C.EXP_RANDOM_P, n: int = C.EXP_RANDOM_N) -> dict:
    """Zufalls-DAGs (Gewichte ±U(0,5; 1,5), Einheitsrauschen): die drei Verfahren auf Rohdaten und standardisierten Daten; Varsortability beider."""
    rng = SplitMix64(C.SEED_BASE)
    per = {sc: {m: [] for m in METHODS} for sc in C.SCALES}
    vs = {sc: [] for sc in C.SCALES}
    used = 0
    for r in range(dags):
        b = W.random_dag(d, p, rng)
        if not b.any():
            continue
        used += 1
        truth = {(i, j) for j in range(d) for i in range(d) if b[j, i] != 0}
        x = W.simulate_linear(b, n, SplitMix64(C.SEED_BASE + 1000 + r))
        for sc, data in ((C.SCALES[0], x), (C.SCALES[1], B.standardize(x))):
            vs[sc].append(B.varsortability(truth, data))
            for m, row in _methods(data, truth, C.SEED_BASE + 1000 + r).items():
                per[sc][m].append(row)
    return {"dags": used, "n": n, "nodes": d, "varsort": {sc: float(np.mean(v)) for sc, v in vs.items()}, "rows": {sc: {m: _aggregate(rows) for m, rows in ms.items()} for sc, ms in per.items()}}


# ----------------------------------------------------------------------------- Experiment 2: die Welt, roh und standardisiert, nach n

def world_experiment(ns=C.EXP_WORLD_NS, reps: int = C.EXP_WORLD_REPS) -> dict:
    world = W.World()
    truth = truth_edges(world)
    rows = []
    for j, n in enumerate(ns):
        for sc in C.SCALES:
            per = {m: [] for m in METHODS}
            sz = zx = full = 0
            for r in range(reps):
                data = simulate_scaled(world, n, SplitMix64(C.SEED_BASE + 100000 + 1000 * j + r), 1.0, sc == C.SCALES[1])
                learned = N.threshold_dag(N.notears(data).w, C.DEFAULT_THR)
                sz += (W.S, W.Z) in learned
                zx += (W.Z, W.X) in learned
                full += learned == truth
                per["NOTEARS"].append(B.compare(learned, set(), truth))
                per["Rangbaseline"].append(B.compare(B.sortnregress(data, tie_seed=C.SEED_BASE + 100000 + 1000 * j + r), set(), truth))
                g = P.pc(5, P.fisher_z_test(np.corrcoef(data, rowvar=False), n, C.PC_ALPHA)).graph
                per["PC"].append(B.compare(set(g[0]), set(g[1]), truth))
            rows.append({"n": n, "scale": sc, "sz": sz / reps, "zx": zx / reps, "full": full / reps, "methods": {m: _aggregate(v) for m, v in per.items()}})
    return {"reps": reps, "rows": rows, "varsort_raw": B.varsortability(truth, simulate_scaled(world, 1000, SplitMix64(1))), "varsort_std": B.varsortability(truth, simulate_scaled(world, 1000, SplitMix64(1), 1.0, True))}


# ----------------------------------------------------------------------------- Experiment 3: Rauschstärke der Wurzel

def sigma_experiment(sigmas=C.EXP_SIGMAS, reps: int = C.EXP_SIGMA_REPS, n: int = C.EXP_SIGMA_N) -> dict:
    world = W.World()
    rows = []
    for i, sigma in enumerate(sigmas):
        nt = rank = 0
        var_s = var_z = 0.0
        for r in range(reps):
            data = simulate_scaled(world, n, SplitMix64(C.SEED_BASE + 200000 + 1000 * i + r), sigma)
            nt += (W.S, W.Z) in N.threshold_dag(N.notears(data).w, C.DEFAULT_THR)
            rank += (W.S, W.Z) in B.sortnregress(data, tie_seed=C.SEED_BASE + 200000 + 1000 * i + r)
            v = data.var(axis=0)
            var_s += v[W.S] / reps
            var_z += v[W.Z] / reps
        rows.append({"sigma": sigma, "notears": nt / reps, "rank": rank / reps, "var_s": var_s, "var_z": var_z, "var_s_exact": sigma ** 2, "var_z_exact": world.a_sz ** 2 * sigma ** 2 + 1.0})
    return {"reps": reps, "n": n, "rows": rows, "flip": C.SIGMA_FLIP}


# ----------------------------------------------------------------------------- Experiment 4: die Nebenbedingung und die festen Zahlen

def constraint_experiment(reps_unconstrained: int = C.EXP_UNCONSTRAINED_REPS, reps_grid: int = C.EXP_GRID_REPS, n: int = C.EXP_GRID_N, lams=C.EXP_GRID_LAMBDAS, thrs=C.EXP_GRID_THRS) -> dict:
    world = W.World()
    truth = truth_edges(world)
    first = N.notears(simulate_scaled(world, n, SplitMix64(C.DEFAULT_SEED)))
    cyclic = 0
    for r in range(reps_unconstrained):
        res = N.notears(simulate_scaled(world, n, SplitMix64(C.SEED_BASE + 300000 + r)), constrained=False)
        cyclic += not N.is_acyclic({(i, j) for i in range(5) for j in range(5) if abs(res.w[i, j]) > C.DEFAULT_THR})
    grid = []
    for k, lam in enumerate(lams):
        ws = [N.notears(simulate_scaled(world, n, SplitMix64(C.SEED_BASE + 400000 + 1000 * k + r)), lam=lam).w for r in range(reps_grid)]
        for thr in thrs:
            rows = [B.compare(N.threshold_dag(w, thr), set(), truth) for w in ws]
            grid.append({"lam": lam, "thr": thr, "shd": float(np.mean([r["shd"] for r in rows])), "exact": float(np.mean([r["shd"] == 0 for r in rows])), "missing": float(np.mean([r["missing"] for r in rows])),
                         "extra": float(np.mean([r["extra"] for r in rows]))})
    return {"trace": first.trace, "h_final": first.h, "cyclic": cyclic / reps_unconstrained, "reps_unconstrained": reps_unconstrained, "reps_grid": reps_grid, "grid": grid}


# ----------------------------------------------------------------------------- Baukasten für h(W)

BUILDER_NODES = ("A", "B", "C")
BUILDER_ARROWS = tuple((i, j) for i in range(3) for j in range(3) if i != j)


def builder_matrix(selected, weight: float) -> np.ndarray:
    w = np.zeros((3, 3))
    for i, j in selected:
        w[i, j] = weight
    return w


def builder_result(selected, weight: float) -> dict:
    w = builder_matrix(selected, weight)
    return {"h": N.h_value(w), "acyclic": N.is_acyclic(set(selected)), "edges": len(list(selected)), "w": w}


# ----------------------------------------------------------------------------- Verdikte

def _name(a: int, b: int) -> str:
    return f"{W.NAMES[a]} → {W.NAMES[b]}"


def verdict_instance(a: Analysis) -> tuple:
    s = a.settings
    nt, rk, pc = a.compare["NOTEARS"], a.compare["Rangbaseline"], a.compare["PC"]
    total = nt["n_true"]
    scale_text = "Standardisiert" if s.standardized else "Roh"
    head = (f"{scale_text}: NOTEARS richtet {nt['correct']} von {total} wahren Kanten richtig und {nt['reversed']} falsch herum (SHD {de(nt['shd'], 1)}); die Rangbaseline, die nur Varianzen sortiert, {rk['correct']} und {rk['reversed']} "
            f"(SHD {de(rk['shd'], 1)}); PC lässt {pc['undirected']} ungerichtet und richtet {pc['correct']} richtig (SHD {de(pc['shd'], 1)}). Varsortability der Daten: {de(a.varsort, 2)}.")
    wrong = [e for e in a.status if e[2] in ("umgekehrt", "überzählig", "fehlt")]
    if nt["shd"] == 0:
        return "success", head + " NOTEARS trifft das wahre Diagramm genau; die Richtungen folgen der Varianzordnung."
    detail = "; ".join(f"{_name(x, y)} {u}" for x, y, u in wrong)
    if s.standardized:
        return "warning", head + f" Nach der Standardisierung haben alle Spalten Varianz 1: Die Richtung bleibt der Rechnung überlassen ({detail})."
    return "warning", head + f" Abweichungen: {detail}."


def verdict_pairs(a: Analysis) -> tuple:
    ties = sum(1 for r in a.pairs if r["prefers"] == "Gleichstand")
    right = sum(1 for r in a.pairs if r["prefers"] == "a → b")
    wrong = sum(1 for r in a.pairs if r["prefers"] == "b → a")
    return "info", (f"Für jedes wahre Paar vergleicht der Verlust beide Richtungen: {right} von {len(a.pairs)} bevorzugen die richtige, {wrong} die falsche, {ties} sind ein Gleichstand. "
                    "Die Differenz ist ½ ρ² (Var(Ursache) − Var(Wirkung)): Der Verlust bevorzugt die Richtung, in der die Größe mit der kleineren Varianz die Ursache ist.")


def verdict_random(res: dict) -> tuple:
    r, s = res["rows"][C.SCALES[0]], res["rows"][C.SCALES[1]]
    return "info", (f"In {res['dags']} Zufalls-DAGs mit {res['nodes']} Knoten (n = {_n(res['n'])}) fällt die Varsortability von {de(res['varsort'][C.SCALES[0]], 2)} auf {de(res['varsort'][C.SCALES[1]], 2)}, wenn man standardisiert. "
                    f"NOTEARS richtet roh {pct(r['NOTEARS']['correct'])} der wahren Kanten richtig (SHD {de(r['NOTEARS']['shd'], 2)}), die Rangbaseline {pct(r['Rangbaseline']['correct'])} (SHD {de(r['Rangbaseline']['shd'], 2)}); "
                    f"standardisiert sind es {pct(s['NOTEARS']['correct'])} und {pct(s['Rangbaseline']['correct'])}, {pct(s['NOTEARS']['reversed'])} und {pct(s['Rangbaseline']['reversed'])} sind umgekehrt. "
                    f"PC ändert sich nicht (SHD {de(r['PC']['shd'], 2)}), weil Korrelationen von der Skala nicht abhängen.")


def verdict_world(res: dict) -> tuple:
    big = max(r["n"] for r in res["rows"])
    raw = next(r for r in res["rows"] if r["n"] == big and r["scale"] == C.SCALES[0])
    std = next(r for r in res["rows"] if r["n"] == big and r["scale"] == C.SCALES[1])
    small = min(r["n"] for r in res["rows"])
    std_small = next(r for r in res["rows"] if r["n"] == small and r["scale"] == C.SCALES[1])
    return "info", (f"In der Welt der Linie (n = {_n(big)}, {res['reps']} Läufe je Zelle) richtet NOTEARS roh S → Z in {pct(raw['sz'])} und Z → X in {pct(raw['zx'])} der Läufe richtig, den ganzen DAG in {pct(raw['full'])}; PC lässt beide Kanten offen "
                    f"(ungerichtet {pct(raw['methods']['PC']['undirected'])} der wahren Kanten). Standardisiert sind es {pct(std['sz'])}, {pct(std['zx'])} und {pct(std['full'])}, {pct(std['methods']['NOTEARS']['reversed'])} der Kanten sind umgekehrt; "
                    f"mit n = {small} nicht anders ({pct(std_small['full'])} ganze DAGs): Zwischen n = {small} und n = {_n(big)} ändert sich daran nichts.")


def verdict_sigma(res: dict) -> tuple:
    rows = res["rows"]
    mid = rows[1]
    last = rows[-1]
    return "info", (f"Die Varianzordnung von Saison und Nachfrage kippt bei σ = {de(res['flip'], 2)} (Var(S) = σ², Var(Z) = 0,64 σ² + 1). Bei σ = {de(mid['sigma'], 1)} (Var(S) {de(mid['var_s'], 2)} gegen Var(Z) {de(mid['var_z'], 2)}) richtet NOTEARS S → Z in "
                    f"{pct(mid['notears'])} der Läufe richtig, die Rangbaseline in {pct(mid['rank'])}; bei σ = {de(last['sigma'], 1)} richtet NOTEARS sie in {pct(last['notears'])}, die Rangbaseline in {pct(last['rank'])} richtig.")


def verdict_constraint(res: dict) -> tuple:
    tr = res["trace"]
    grid = {(r["lam"], r["thr"]): r for r in res["grid"]}
    return "info", (f"Das augmentierte Lagrange-Verfahren braucht {len(tr)} Teilprobleme: h(W) fällt von {de(tr[0]['h'], 3)} auf {de(tr[-1]['h'] * 1e5, 1)}·10⁻⁵, der Verlust steigt von {de(tr[0]['loss'], 3)} auf {de(tr[-1]['loss'], 3)} "
                    f"(die Nebenbedingung kostet Anpassung). Ohne sie ist der Graph in {pct(res['cyclic'])} der {res['reps_unconstrained']} Läufe zyklisch. Die festen Zahlen wirken zusammen: bei λ = 0,1 trifft die Schwelle 0,3 "
                    f"(SHD {de(grid[(0.1, 0.3)]['shd'], 2)}), bei λ = 0,3 nicht mehr (SHD {de(grid[(0.3, 0.3)]['shd'], 2)}).")


def verdict_builder(b: dict) -> tuple:
    if b["edges"] == 0:
        return "info", "Ohne Kante ist W null und h(W) = 0."
    if b["acyclic"]:
        return "success", f"Kein gerichteter Kreis: h(W) = {de(b['h'], 4)} (null bis auf Rundung). Genau dann ist W die Gewichtsmatrix eines DAG."
    return "error", f"Ein gerichteter Kreis: h(W) = {de(b['h'], 4)} ist positiv; je stärker die Gewichte auf dem Kreis, desto größer."
