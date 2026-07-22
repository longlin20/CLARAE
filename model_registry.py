"""
model_registry.py — Registro central de modelos.

Para añadir un modelo nuevo edita ÚNICAMENTE la tabla _REGISTRY.
Todos los scripts (run_training, run_test, run_experiments, eval_utils, …)
importan desde aquí: ningun otro fichero necesita cambios.

Columnas de cada entrada:
  name          – clave string usada en todos los scripts
  module        – ruta Python del módulo de arquitectura
  cls_name      – nombre de la clase dentro del módulo
  family        – grupo para --families en run_experiments.py
  is_sc         – el encoder devuelve (latent, skips)  [→ SC_MODELS]
  is_clarae     – usa constructor estilo CLARAE          [→ _CLARAE_MODELS]
  has_skip_drop – acepta kwarg skip_dropout              [→ SKIP_DROP_MODELS]
"""

import re
from collections import namedtuple

_E = namedtuple('_E', ['name', 'module', 'cls_name', 'family',
                        'is_sc', 'is_clarae', 'has_skip_drop'])

# =============================================================================
# MASTER TABLE — editar solo aquí para añadir/eliminar modelos
# =============================================================================
_REGISTRY = [
    # ── #1: Baseline simple (single-scale, AvgPool, ELU, no skips) ───────────
    _E("CLARAE_AP_ELU",                           "architecture.autoencoders_improved", "CLARAE_AP_ELU",                           "AP_ELU",        False, True,  False),
    # ── #2/#3: SCM + GLU + AP + ELU — sin/con IN en las skips ───────────────
    _E("CLARAE_SCM_GLU_AP_ELU_woIN",              "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_woIN",              "SCM_woIN",      True,  True,  False),
    _E("CLARAE_SCM_GLU_AP_ELU",                   "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU",                   "SCM_ACT",       True,  True,  False),
    _E("CLARAE_SCM_AP_ELU",                       "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU",                       "SCM_noGLU",           True,  True,  False),
    _E("CLARAE_SCM_AP_ELU_woIN",                  "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_woIN",                  "SCM_noGLU_woIN",      True,  True,  False),
    _E("CLARAE_SCM_AP_ELU_K2",                    "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_K2",                    "SCM_noGLU_K2",        True,  True,  False),
    # ── #4/#5: SCM + GLU + AP + ELU + gate skip3+skip4 — sin/con IN ─────────
    _E("CLARAE_SCM_GLU_AP_ELU_GATE_woIN",         "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_GATE_woIN",         "SCM_GATE_woIN",       True,  True,  True),
    _E("CLARAE_SCM_GLU_AP_ELU_GATE",              "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_GATE",              "SCM_GATE",            True,  True,  True),
    _E("CLARAE_SCM_AP_ELU_GATE",                  "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_GATE",                  "SCM_GATE_noGLU",      True,  True,  True),
    _E("CLARAE_SCM_AP_ELU_GATE_K2",               "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_GATE_K2",               "SCM_GATE_noGLU_K2",   True,  True,  True),
    # ── #7a: SCM + GLU + AP + ELU + gate skip4 only — kernels (3,9,21) ──────
    _E("CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST",       "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST",       "SCM_SK",           True,  True,  True),
    _E("CLARAE_SCM_AP_ELU_GATE_SKLAST",           "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_GATE_SKLAST",           "SCM_SK_noGLU",          True,  True,  True),
    _E("CLARAE_SCM_AP_ELU_SKLAST",               "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_SKLAST",               "SCM_SK_noGLU_noGate",   True,  True,  True),
    # ── #7b: SKLAST con kernels (3,9) únicamente ─────────────────────────────
    _E("CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_K2",    "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_K2",    "SCM_SK_K2",             True,  True,  True),
    _E("CLARAE_SCM_AP_ELU_GATE_SKLAST_K2",        "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_GATE_SKLAST_K2",        "SCM_SK_K2_noGLU",       True,  True,  True),
    _E("CLARAE_SCM_AP_ELU_SKLAST_K2",             "architecture.autoencoders_improved", "CLARAE_SCM_AP_ELU_SKLAST_K2",             "SCM_SK_K2_noGLU_noGate",True,  True,  True),
    # ── #8: GATE en las 4 capas ──────────────────────────────────────────────
    _E("CLARAE_SCM_GLU_AP_ELU_GATE_SKALL",        "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_GATE_SKALL",        "SCM_GATE_ALL",     True,  True,  True),
    # ── #9: Ablaciones SKLAST — sin gate/IN y sin SCM ────────────────────────
    _E("CLARAE_SCM_GLU_AP_ELU_SKLAST_K2",          "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_SKLAST_K2",          "SCM_SK_K2_noGate", True,  True,  True),
    _E("CLARAE_GLU_AP_ELU_GATE_SKLAST",           "architecture.autoencoders_improved", "CLARAE_GLU_AP_ELU_GATE_SKLAST",           "GLU_GATE_SK_noSCM", True,  True,  True),
    _E("CLARAE_AP_ELU_GATE_SKLAST",              "architecture.autoencoders_improved", "CLARAE_AP_ELU_GATE_SKLAST",              "ELU_GATE_SK_noSCM", True,  True,  True),
    # ── Ablaciones del mejor modelo ───────────────────────────────────────────
    _E("CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_TANH",  "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_GATE_SKLAST_TANH",  "SCM_SK_TANH",   True,  True,  True),

    _E("CLARAE_SCM_GLU_AP_ELU_SKALL",             "architecture.autoencoders_improved", "CLARAE_SCM_GLU_AP_ELU_SKALL",             "SCM_SKALL_AP_noGate",  True,  True,  True),
    # ── BASELINE ─────────────────────────────────────────────────────────────
    _E("ACDAE",       "architecture.acdae",       "ACDAE",       "BASELINE", False, False, False),
    _E("FGDAE",       "architecture.fgdae",       "FGDAE",       "BASELINE", False, False, False),
    _E("DRNN",        "architecture.drnn",        "DRNN",        "BASELINE", False, False, False),
    _E("CNN-DAE",     "architecture.dae",         "CNN_DAE",     "BASELINE", False, False, False),
    _E("FCN-DAE",     "architecture.dae",         "FCN_DAE",     "BASELINE", False, False, False),
    _E("DEEP-FILTER", "architecture.deepfilter",  "DeepFilter",  "BASELINE", False, False, False),
]

# =============================================================================
# Estructuras derivadas — NO editar manualmente
# =============================================================================

# eval_utils / latent_space_analysis: {name: (module, cls_name)}
MODEL_REGISTRY = {e.name: (e.module, e.cls_name) for e in _REGISTRY}

# run_test: {name: {"model_architecture": name}}
MODEL_REGISTRY_TEST = {e.name: {"model_architecture": e.name} for e in _REGISTRY}

# run_experiments: [(arch_key, model_architecture, sufijo, family_tag)]
ALL_VARIANTS = [(e.name, e.name, e.name, e.family) for e in _REGISTRY]

# run_training argparse choices
ALL_MODEL_NAMES = [e.name for e in _REGISTRY]

# Sets por propiedad
SC_MODELS        = {e.name for e in _REGISTRY if e.is_sc}
_CLARAE_MODELS   = {e.name for e in _REGISTRY if e.is_clarae}
SKIP_DROP_MODELS = {e.name for e in _REGISTRY if e.has_skip_drop}
# Models with learned sigmoid gates (accept gate_init kwarg)
GATE_MODELS      = {e.name for e in _REGISTRY if e.has_skip_drop and 'GATE' in e.name}
ENCODE_MODELS    = {'ACDAE', 'FCN-DAE', 'CNN-DAE', 'FGDAE'}
# Models that do NOT support latent space extraction (skip clf & t-SNE)
NO_LATENT_MODELS = {'DRNN', 'ACDAE', 'DEEP-FILTER'}

# Sufijos de loss/noise (compartidos por run_test y eval_utils)
_KNOWN_LOSSES     = ["dtw", "mse"]
_KNOWN_NOISE_TAGS = ["noNoise", "noise"]


def _strip_loss_noise_suffix(arch_key: str):
    """Elimina el sufijo _{loss}_{noise_tag}[_skdXX][_pwYY][_klXX] del nombre de arquitectura."""
    arch_key = re.sub(r'_kl\d+$', '', arch_key)
    arch_key = re.sub(r'_pw\d+$', '', arch_key)
    arch_key = re.sub(r'_skd\d{2}$', '', arch_key)
    for noise in _KNOWN_NOISE_TAGS:
        if arch_key.endswith(f"_{noise}"):
            rest = arch_key[:-(len(noise) + 1)]
            for loss in _KNOWN_LOSSES:
                if rest.endswith(f"_{loss}"):
                    return rest[:-(len(loss) + 1)], loss, noise
    return arch_key, None, None


def get_model_class(name: str):
    """Importa y devuelve la clase del modelo por su nombre en el registro."""
    import importlib
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"'{name}' no está en MODEL_REGISTRY.\n"
            f"  Disponibles: {list(MODEL_REGISTRY.keys())}"
        )
    mod_path, cls_name = MODEL_REGISTRY[name]
    return getattr(importlib.import_module(mod_path), cls_name)
