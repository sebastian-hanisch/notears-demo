"""Gemeinsame Fixtures: Hand-Instanzen (DEMO-PLAYBOOK §1b) und die Experimente, einmal klein (für Struktur- und Verdikt-Tests) und einmal in voller Größe (nur für test_claims)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def chain_cov():
    """Mini-Instanz: x1 ~ N(0, 1), x2 = 0,8 x1 + ε mit ε ~ N(0, 1): Kovarianz [[1, 0,8], [0,8, 1,64]]."""
    return np.array([[1.0, 0.8], [0.8, 1.64]])


@pytest.fixture(scope="session")
def small_random():
    import nts_evaluation as ev

    return ev.random_experiment(dags=3, n=500)


@pytest.fixture(scope="session")
def small_world():
    import nts_evaluation as ev

    return ev.world_experiment(ns=(300,), reps=2)


@pytest.fixture(scope="session")
def small_sigma():
    import nts_evaluation as ev

    return ev.sigma_experiment(sigmas=(1.0, 1.5, 2.0), reps=3, n=500)


@pytest.fixture(scope="session")
def small_constraint():
    import nts_evaluation as ev

    return ev.constraint_experiment(reps_unconstrained=3, reps_grid=2, n=500, lams=(0.1, 0.3), thrs=(0.3, 0.5))


@pytest.fixture(scope="session")
def full_random():
    import nts_evaluation as ev

    return ev.random_experiment()


@pytest.fixture(scope="session")
def full_world():
    import nts_evaluation as ev

    return ev.world_experiment()


@pytest.fixture(scope="session")
def full_sigma():
    import nts_evaluation as ev

    return ev.sigma_experiment()


@pytest.fixture(scope="session")
def full_constraint():
    import nts_evaluation as ev

    return ev.constraint_experiment()
