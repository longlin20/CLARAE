"""
run_experiments.py
Ejecuta modelos de múltiples familias secuencialmente con los mismos
hiperparámetros, garantizando comparación justa (misma semilla de ruido en test).

Uso:
    python run_experiments.py                              # todos los modelos
    python run_experiments.py --dry_run                    # imprime comandos sin ejecutar
    python run_experiments.py --families SCM BASELINE      # solo esas familias
    python run_experiments.py --models CLARAE_SCM_GLU ACDAE  # solo esos modelos
"""

import subprocess
import sys
import os
import argparse
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIGURACIÓN COMÚN
# Modifica estos valores antes de lanzar el experimento.
# =============================================================================

COMMON = {
    "preprocessed_data_dir": "processed_data_final/",
    "epochs":                150,
    "batch_size":            256,
    "learning_rate":         0.0002,
    "optimizer":             "Adam",
    "weight_decay":          1e-3,
    "latent_dim":            64,
    "filters_initial":       64,
    "dropout_rate":          0.1,
    "skip_dropout":          0.7,
    "gate_init":             -5.0,
    "dense_dim":             64,
    "loss_function":         "dtw",
    "patience_lr":           3,
    "factor_lr":             0.5,
    "patience_es":           9,
    # FGDAE específico
    "q_parameter":           2,
    # Ruido en entrenamiento
    "add_noise":             True,
    "random_noise":          True,
    "noise_snr_min":         -5.0,
    "noise_snr_max":         10.0,
    "min_noise_types":       1,
    "max_noise_types":       4,
    "powerline_freq":        50,
    "sampling_freq":         500,
    # VAE
    "kl_beta":               0.0,
    # Reproducibilidad
    "seed":                  42,
    "noise_seed":            0,
    # WandB
    "wandb_project":         "autoencoder-egms-new",
    # Dispositivo
    "device":                "cuda",
}

# =============================================================================
# VARIANTES POR FAMILIA
# Cada entrada: (arch_key, model_architecture, sufijo_model_name, familia [, overrides])
#   arch_key          → identificador usado en --models
#   model_architecture → nombre exacto que acepta run_training.py
#   sufijo_model_name  → sufijo del nombre de fichero guardado (se añade al model_name)
#   familia           → grupo usado en --families
#   overrides (opt.)  → dict con claves de COMMON a sobreescribir para esta variante
#
# Ejemplos de overrides:
#   {"loss_function": "mse"}
#   {"add_noise": False, "random_noise": False}
#   {"epochs": 50, "patience_lr": 5, "patience_es": 10}
# =============================================================================

from model_registry import ALL_VARIANTS  # noqa: E402

# NOTE: para añadir un modelo nuevo edita model_registry.py, no este fichero.
# Si una variante necesita overrides de COMMON, añade un 5.º elemento dict a su
# entrada en ALL_VARIANTS (que en ese caso deberás definir aquí manualmente).

ALL_FAMILIES = sorted({v[3] for v in ALL_VARIANTS})
ALL_KEYS     = [v[0] for v in ALL_VARIANTS]

def build_cmd(arch: str, model_name: str, common: dict) -> list[str]:
    """Construye la lista de argumentos para run_training.py."""
    cmd = [sys.executable, os.path.join(_SCRIPT_DIR, "run_training.py")]

    # Obligatorios
    cmd += ["--preprocessed_data_dir", common["preprocessed_data_dir"]]
    cmd += ["--model_architecture", arch]
    signal_type = "UNIPOLAR" if common.get("unipolar", True) else "BIPOLAR"
    cmd += ["--model_name", f"{signal_type}_{model_name}"]

    # Tipo de señal
    if common.get("unipolar", True):
        cmd += ["--unipolar"]
    else:
        cmd += ["--bipolar"]

    # Hiperparámetros de entrenamiento
    cmd += ["--epochs",          str(common["epochs"])]
    cmd += ["--batch_size",      str(common["batch_size"])]
    cmd += ["--learning_rate",   str(common["learning_rate"])]
    cmd += ["--optimizer",       common["optimizer"]]
    cmd += ["--weight_decay",    str(common["weight_decay"])]
    cmd += ["--latent_dim",      str(common["latent_dim"])]
    cmd += ["--filters_initial", str(common["filters_initial"])]
    cmd += ["--dropout_rate",    str(common["dropout_rate"])]
    cmd += ["--skip_dropout",          str(common["skip_dropout"])]
    cmd += ["--gate_init",             str(common["gate_init"])]
    cmd += ["--dense_dim",             str(common["dense_dim"])]

    # Loss
    cmd += ["--loss_function", common["loss_function"]]
    # LR scheduler
    cmd += ["--patience_lr", str(common["patience_lr"])]
    cmd += ["--factor_lr",   str(common["factor_lr"])]

    # Early stopping
    cmd += ["--patience_es", str(common["patience_es"])]

    # FGDAE específico
    cmd += ["--q_parameter", str(common["q_parameter"])]

    # VAE
    if common.get("kl_beta", 0.0) > 0:
        cmd += ["--kl_beta", str(common["kl_beta"])]

    # Ruido
    if common.get("add_noise"):
        cmd += ["--add_noise"]
    if common.get("random_noise"):
        cmd += ["--random_noise"]
    cmd += ["--noise_snr_min",    str(common["noise_snr_min"])]
    cmd += ["--noise_snr_max",    str(common["noise_snr_max"])]
    cmd += ["--min_noise_types",  str(common["min_noise_types"])]
    cmd += ["--max_noise_types",  str(common["max_noise_types"])]
    cmd += ["--powerline_freq",   str(common["powerline_freq"])]
    cmd += ["--sampling_freq",    str(common["sampling_freq"])]

    # Reproducibilidad
    cmd += ["--seed",       str(common["seed"])]
    cmd += ["--noise_seed", str(common["noise_seed"])]

    # WandB
    cmd += ["--wandb_project", common["wandb_project"]]

    # Dispositivo
    cmd += ["--device", common["device"]]

    # Filtro de potencia DF
    if common.get("min_power", 0.0) > 0.0:
        cmd += ["--min_power", str(common["min_power"])]

    return cmd

def main():
    parser = argparse.ArgumentParser(
        description="Lanza experimentos de múltiples familias de modelos secuencialmente"
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Imprime los comandos sin ejecutarlos")
    signal_group = parser.add_mutually_exclusive_group()
    signal_group.add_argument("--unipolar", action="store_true", default=False,
                              help="Usar señales unipolares (default: bipolar)")
    signal_group.add_argument("--bipolar", action="store_true", default=False,
                              help="Usar señales bipolares (default)")
    parser.add_argument("--families", nargs="+", default=None,
                        choices=ALL_FAMILIES,
                        help=f"Familias a ejecutar: {ALL_FAMILIES} (default: todas)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=ALL_KEYS,
                        help="Modelos concretos a ejecutar por arch_key (default: todos)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Ruta al directorio de datos preprocesados (sobreescribe COMMON)")
    # Overrides de COMMON por CLI (se aplican a todos los modelos del run)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Dimensión del espacio latente (sobreescribe COMMON['latent_dim'])")
    parser.add_argument("--filters_initial", type=int, default=None,
                        help="Número inicial de filtros (sobreescribe COMMON['filters_initial'])")
    parser.add_argument("--dense_dim", type=int, default=None,
                        help="Dimensión de la capa densa (sobreescribe COMMON['dense_dim'])")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Número de épocas de entrenamiento (sobreescribe COMMON['epochs'])")
    # Si se pasan varios valores, se ejecuta el producto cartesiano de combinaciones.
    parser.add_argument("--loss_functions", nargs="+", default=None,
                        choices=["mse", "dtw"],
                        help="Una o varias loss functions a barrer (genera combinaciones)")
    parser.add_argument("--noise_modes", nargs="+", default=None,
                        choices=["noise", "no_noise"],
                        help="Uno o varios modos de ruido a barrer: noise no_noise")
    parser.add_argument("--skip_dropout", type=float, default=None,
                        help="Tasa de dropout para skip connections (sobreescribe COMMON['skip_dropout']). "
                             "Añade sufijo _skdXX al nombre del modelo.")
    parser.add_argument("--gate_init", type=float, default=None,
                        help="Logit inicial de los gates de skip (sobreescribe COMMON['gate_init']). "
                             "Más negativo = gates se abren más lento.")
    parser.add_argument("--patience_es", type=int, default=None,
                        help="Paciencia para early stopping (sobreescribe COMMON['patience_es']).")
    parser.add_argument("--patience_lr", type=int, default=None,
                        help="Paciencia para reducción de LR (sobreescribe COMMON['patience_lr']).")
    parser.add_argument("--kl_beta", type=float, default=None,
                        help="Peso de la pérdida KL en modelos VAE (sobreescribe COMMON['kl_beta']).")
    parser.add_argument("--min_power", type=float, default=None,
                        help="Umbral sum_power DF: excluye señales con sum_power < min_power "
                             "de train y val. Añade sufijo _pwXX al nombre del modelo.")
    args = parser.parse_args()

    # Tipo de señal: --unipolar activa unipolar, por defecto bipolar
    COMMON["unipolar"] = args.unipolar
    if args.data_dir is not None:
        COMMON["preprocessed_data_dir"] = args.data_dir
    if args.latent_dim is not None:
        COMMON["latent_dim"] = args.latent_dim
    if args.filters_initial is not None:
        COMMON["filters_initial"] = args.filters_initial
    if args.dense_dim is not None:
        COMMON["dense_dim"] = args.dense_dim
    if args.epochs is not None:
        COMMON["epochs"] = args.epochs
    if args.skip_dropout is not None:
        COMMON["skip_dropout"] = args.skip_dropout
        COMMON["skip_dropout_suffix"] = f"_skd{int(args.skip_dropout * 100):02d}"
    else:
        COMMON["skip_dropout_suffix"] = ""
    if args.gate_init is not None:
        COMMON["gate_init"] = args.gate_init
    if args.patience_es is not None:
        COMMON["patience_es"] = args.patience_es
    if args.patience_lr is not None:
        COMMON["patience_lr"] = args.patience_lr
    if args.kl_beta is not None:
        COMMON["kl_beta"] = args.kl_beta
        COMMON["kl_suffix"] = f"_kl{str(args.kl_beta).replace('.', '')}"
    else:
        COMMON["kl_suffix"] = ""
    if args.min_power is not None:
        COMMON["min_power"] = args.min_power
        COMMON["min_power_suffix"] = f"_pw{int(args.min_power * 100):02d}"
    else:
        COMMON["min_power"] = 0.0
        COMMON["min_power_suffix"] = ""
    COMMON["q_suffix"] = ""

    # Construir lista de combinaciones (loss × noise)
    # Cada combo: (overrides_dict, sufijo_str)
    loss_list  = args.loss_functions or [None]   # None = usar COMMON
    noise_list = args.noise_modes    or [None]   # None = usar COMMON

    sweep_combos = []
    for lf in loss_list:
        for nm in noise_list:
            combo_overrides = {}
            # Loss: siempre incluir en el nombre (efectivo o COMMON)
            effective_lf = lf if lf is not None else COMMON["loss_function"]
            if lf is not None:
                combo_overrides["loss_function"] = lf
            # Noise: siempre incluir en el nombre
            if nm == "no_noise":
                combo_overrides["add_noise"]    = False
                combo_overrides["random_noise"] = False
                noise_tag = "noNoise"
            elif nm == "noise":
                combo_overrides["add_noise"] = True
                noise_tag = "noise"
            else:  # None → usar COMMON
                noise_tag = "noise" if COMMON.get("add_noise") else "noNoise"
            suffix = f"_{effective_lf}_{noise_tag}"
            sweep_combos.append((combo_overrides, suffix))

    # Filtrar variantes
    variants = ALL_VARIANTS
    if args.families:
        variants = [v for v in variants if v[3] in args.families]
    if args.models:
        variants = [v for v in variants if v[0] in args.models]

    total_runs = len(variants) * len(sweep_combos)

    # Agrupar por familia para el resumen de cabecera
    families_in_run = {}
    for v in variants:
        families_in_run.setdefault(v[3], []).append(v[1])

    print("=" * 70)
    print(f"  Experiment suite  -  {len(variants)} modelo(s) × {len(sweep_combos)} combo(s) = {total_runs} run(s)")
    for fam, archs in families_in_run.items():
        print(f"    [{fam}] {', '.join(archs)}")
    if len(sweep_combos) > 1:
        print(f"  Sweep: loss={loss_list}  noise={noise_list}")
    print("=" * 70)

    results = []

    for v in variants:
        key, arch, name, family = v[:4]
        variant_overrides = v[4] if len(v) > 4 else {}

        for combo_overrides, combo_suffix in sweep_combos:
            # Prioridad: COMMON < combo < variante
            config = {**COMMON, **combo_overrides, **variant_overrides}

            # Nombre único por combinación (incluye sufijo skd/kl si se pasaron)
            run_name = name + combo_suffix + config.get("skip_dropout_suffix", "") + config.get("kl_suffix", "") + config.get("q_suffix", "") + config.get("min_power_suffix", "")

            cmd = build_cmd(arch, run_name, config)

            print(f"\n{'-'*70}")
            print(f"  Familia: {family}  |  Modelo: {arch}{combo_suffix}")
            if combo_overrides:
                print(f"  Combo:    {combo_overrides}")
            if variant_overrides:
                print(f"  Variante: {variant_overrides}")
            print(f"  Cmd:    {' '.join(cmd)}")
            print(f"{'-'*70}")

            if args.dry_run:
                print("  [dry_run] Omitiendo ejecución.")
                continue

            t0 = time.time()
            ret = subprocess.run(cmd)
            elapsed = time.time() - t0

            status = "OK" if ret.returncode == 0 else f"ERROR (code {ret.returncode})"
            results.append((family, f"{arch}{combo_suffix}", status, elapsed))
            print(f"\n  [{arch}{combo_suffix}] {status}  ({elapsed/60:.1f} min)")

    if not args.dry_run and results:
        print("\n" + "=" * 70)
        print("  RESUMEN")
        print("=" * 70)
        current_family = None
        for family, arch, status, elapsed in results:
            if family != current_family:
                print(f"\n  -- {family} --")
                current_family = family
            print(f"    {arch:<24} {status:<20} {elapsed/60:.1f} min")
        total = sum(e for _, _, _, e in results)
        print(f"\n  Total: {total/60:.1f} min")

if __name__ == "__main__":
    main()