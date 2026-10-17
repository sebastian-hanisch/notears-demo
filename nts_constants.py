"""Konstanten: Regler-Grenzen, feste Experimente, Farben und Presets."""

SEED_MAX, DEFAULT_SEED = 999999, 35

SCALES = ("roh", "standardisiert")
DEFAULT_SCALE = SCALES[0]
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.5, 3.0, 0.25, 1.0       # Streuung des Rauschens der Wurzel S (Einheitsrauschen: 1)
N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 3000, 100, 1000
LAMBDAS = (0.01, 0.1, 0.3)
DEFAULT_LAMBDA = 0.1
THR_MIN, THR_MAX, THR_STEP, DEFAULT_THR = 0.1, 0.6, 0.05, 0.3
PC_ALPHA = 0.01
SIGMA_FLIP = 1.0 / 0.6      # Var(S) = σ² gleich Var(Z) = 0,64 σ² + 1 bei σ = 1/0,6

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 1040000
EXP_RANDOM_DAGS, EXP_RANDOM_NODES, EXP_RANDOM_P, EXP_RANDOM_N = 20, 6, 0.4, 1000        # Experiment 1
EXP_WORLD_NS, EXP_WORLD_REPS = (300, 1000), 16                                          # Experiment 2
EXP_SIGMAS, EXP_SIGMA_REPS, EXP_SIGMA_N = (1.0, 1.5, 2.0), 20, 1000                     # Experiment 3
EXP_UNCONSTRAINED_REPS, EXP_GRID_REPS, EXP_GRID_N = 30, 16, 1000                        # Experiment 4
EXP_GRID_LAMBDAS, EXP_GRID_THRS = (0.01, 0.1, 0.3), (0.1, 0.3, 0.5)

COLOR_TRUTH = "#D68A2E"
COLOR_GOOD = "#2F6B65"
COLOR_BAD = "#c0392b"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_CONTROL = "#4c78a8"


def preset(**kw) -> dict:
    base = dict(scale=DEFAULT_SCALE, sigma=DEFAULT_SIGMA, n=DEFAULT_N, lam=DEFAULT_LAMBDA, thr=DEFAULT_THR, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Rohdaten (Voreinstellung)": preset(),
    "Standardisiert": preset(scale=SCALES[1]),
    "Wurzel knapp": preset(sigma=1.5),
    "Wurzel streut stark": preset(sigma=2.0),
    "Wenige Standorte": preset(n=100, seed=42),
    "Starke Strafe": preset(lam=0.3),
    "Hohe Schwelle": preset(thr=0.5),
}
PRESET_HELP = {
    "Rohdaten (Voreinstellung)": "Die Welt der Linie, Einheitsrauschen, Rohdaten: Die Varianz wächst längs der Pfeile, NOTEARS richtet alle fünf Kanten richtig, auch die, die PC offen lässt.",
    "Standardisiert": "Dieselben Daten, jede Spalte auf Mittel 0 und Streuung 1: Die Varianzordnung ist weg, und NOTEARS richtet die Kanten falsch herum.",
    "Wurzel knapp": "Das Rauschen der Saison ist 1,5-fach: Ihre Varianz (2,25) liegt knapp unter der der Nachfrage (2,37). Die Rangbaseline richtet noch richtig, NOTEARS kippt oft.",
    "Wurzel streut stark": "Das Rauschen der Saison ist 2-fach: Die Saison hat mehr Varianz als die Nachfrage und diese mehr als der Faktor. Die Verfahren richten S–Z und Z–X falsch herum oder verlieren eine Kante.",
    "Wenige Standorte": "Nur 100 Standorte: In dieser Stichprobe fehlt die Kante Nachfrage → Servicegrad.",
    "Starke Strafe": "λ = 0,3 schrumpft die Gewichte stark (wahr 0,6, geschätzt 0,28 bei Wetter → Servicegrad): Die feste Schwelle 0,3 schneidet diese Kante ab.",
    "Hohe Schwelle": "Schwelle 0,5 bei λ = 0,1: Das geschrumpfte Gewicht 0,48 von Wetter → Servicegrad (wahr 0,6) fällt unter die Schwelle.",
}
