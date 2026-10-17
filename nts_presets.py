"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio). Alle Regler sind in jeder Einstellung wirksam und werden immer gezeichnet;
es gibt weder KEPT-Regler noch berechnete Grenzen. Auswahlen (Skala, λ) stehen in der Adresszeile als Index."""

import random
from dataclasses import dataclass

import streamlit as st

import nts_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float
    hi: float
    step: float
    choices: tuple | None = None


def _choice(param: str, choices: tuple, default) -> SettingSpec:
    return SettingSpec(param, int, default, 0, len(choices) - 1, 1, choices)


SETTING_SPECS = {
    "scale_radio": _choice("sc", C.SCALES, C.DEFAULT_SCALE),
    "sigma_slider": SettingSpec("ss", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX, C.SIGMA_STEP),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX, C.N_STEP),
    "lam_slider": _choice("l", C.LAMBDAS, C.DEFAULT_LAMBDA),
    "thr_slider": SettingSpec("th", float, C.DEFAULT_THR, C.THR_MIN, C.THR_MAX, C.THR_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
}
PRESET_KEYS = {"scale": "scale_radio", "sigma": "sigma_slider", "n": "n_slider", "lam": "lam_slider", "thr": "thr_slider", "seed": "seed_input"}


def snap(value, spec: SettingSpec):
    """Auf das Raster des Reglers legen und in die Grenzen klemmen (bei einer Auswahl: auf einen gültigen Index)."""
    value = max(spec.lo, min(spec.hi, value))
    if spec.step > 0 and (spec.caster is float or spec.step > 1 or spec.choices):
        value = round((value - spec.lo) / spec.step) * spec.step + spec.lo
        value = round(value, 6) if spec.caster is float else int(value)
    return spec.choices[int(value)] if spec.choices else value


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    init_session_state_defaults()
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = snap(spec.caster(float(qp[spec.url_param])), spec)
        except (ValueError, TypeError):
            continue
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def url_value(state_key: str, value):
    spec = SETTING_SPECS[state_key]
    return spec.choices.index(value) if spec.choices else value


def sync_query_params(values: dict):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(url_value(state_key, value))
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
