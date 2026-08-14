"""
03_evaluate_ensemble.py

Membandingkan:
1. MLP
2. VGAM
3. Weighted soft-voting ensemble MLP-VGAM

Bobot ensemble dan threshold per label dipilih hanya dari validation set.

Jalankan setelah:
    python 01_prepare_train_mlp.py
    Rscript 02_train_vgam.R
    python 03_evaluate_ensemble.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
)


OUTPUT_DIR = Path("outputs/modeling")
ID_COL = "id"
LABEL_COLS = ["miner", "rust", "phoma"]

MLP_VAL = OUTPUT_DIR / "mlp_val_predictions.csv"
MLP_TEST = OUTPUT_DIR / "mlp_test_predictions.csv"
VGAM_VAL = OUTPUT_DIR / "vgam_val_predictions.csv"
VGAM_TEST = OUTPUT_DIR / "vgam_test_predictions.csv"


def read_predictions(path: Path, model_prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File belum tersedia: {path}")

    df = pd.read_csv(path, dtype={ID_COL: str})

    required = [ID_COL]
    required += [f"true_{label}" for label in LABEL_COLS]
    required += [f"prob_{label}" for label in LABEL_COLS]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak lengkap pada {path}: {missing}")

    rename_map = {
        f"prob_{label}": f"{model_prefix}_prob_{label}"
        for label in LABEL_COLS
    }

    return df.rename(columns=rename_map)


def merge_models(
    mlp_path: Path,
    vgam_path: Path,
) -> pd.DataFrame:
    mlp = read_predictions(mlp_path, "mlp")
    vgam = read_predictions(vgam_path, "vgam")

    true_cols = [f"true_{label}" for label in LABEL_COLS]

    merged = mlp.merge(
        vgam[[ID_COL] + true_cols + [
            f"vgam_prob_{label}" for label in LABEL_COLS
        ]],
        on=ID_COL,
        how="inner",
        suffixes=("_mlp", "_vgam"),
        validate="one_to_one",
    )

    for label in LABEL_COLS:
        left = merged[f"true_{label}_mlp"].to_numpy()
        right = merged[f"true_{label}_vgam"].to_numpy()

        if not np.array_equal(left, right):
            raise ValueError(f"Label true_{label} MLP dan VGAM tidak sama.")

        merged[f"true_{label}"] = left
        merged.drop(
            columns=[f"true_{label}_mlp", f"true_{label}_vgam"],
            inplace=True,
        )

    if len(merged) != len(mlp) or len(merged) != len(vgam):
        raise ValueError("Sebagian ID MLP dan VGAM tidak cocok.")

    return merged


def get_arrays(df: pd.DataFrame):
    y_true = df[
        [f"true_{label}" for label in LABEL_COLS]
    ].to_numpy(dtype=int)

    mlp_prob = df[
        [f"mlp_prob_{label}" for label in LABEL_COLS]
    ].to_numpy(dtype=float)

    vgam_prob = df[
        [f"vgam_prob_{label}" for label in LABEL_COLS]
    ].to_numpy(dtype=float)

    return y_true, mlp_prob, vgam_prob


def find_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> np.ndarray:
    thresholds = []

    for i in range(y_true.shape[1]):
        best_threshold = 0.50
        best_f1 = -1.0

        for threshold in np.arange(0.10, 0.91, 0.01):
            y_pred = (y_prob[:, i] >= threshold).astype(int)
            score = f1_score(
                y_true[:, i],
                y_pred,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(threshold, 2))

        thresholds.append(best_threshold)

    return np.asarray(thresholds)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict, np.ndarray]:
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)

    metrics = {
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "samples_f1": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }

    try:
        metrics["macro_auprc"] = float(
            average_precision_score(y_true, y_prob, average="macro")
        )
        metrics["micro_auprc"] = float(
            average_precision_score(y_true, y_prob, average="micro")
        )
    except ValueError:
        metrics["macro_auprc"] = None
        metrics["micro_auprc"] = None

    return metrics, y_pred


def tune_model(
    y_val: np.ndarray,
    val_prob: np.ndarray,
) -> tuple[np.ndarray, float]:
    thresholds = find_best_thresholds(y_val, val_prob)
    metrics, _ = calculate_metrics(y_val, val_prob, thresholds)
    return thresholds, metrics["macro_f1"]


def tune_ensemble(
    y_val: np.ndarray,
    mlp_prob: np.ndarray,
    vgam_prob: np.ndarray,
):
    best = None

    # alpha = bobot MLP; (1-alpha) = bobot VGAM.
    for alpha in np.arange(0.0, 1.001, 0.05):
        ensemble_prob = alpha * mlp_prob + (1.0 - alpha) * vgam_prob
        thresholds = find_best_thresholds(y_val, ensemble_prob)
        metrics, _ = calculate_metrics(y_val, ensemble_prob, thresholds)

        candidate = {
            "alpha_mlp": float(round(alpha, 2)),
            "alpha_vgam": float(round(1.0 - alpha, 2)),
            "thresholds": thresholds,
            "validation_macro_f1": metrics["macro_f1"],
        }

        if best is None or (
            candidate["validation_macro_f1"]
            > best["validation_macro_f1"]
        ):
            best = candidate

    return best


def main() -> None:
    val_df = merge_models(MLP_VAL, VGAM_VAL)
    test_df = merge_models(MLP_TEST, VGAM_TEST)

    y_val, mlp_val_prob, vgam_val_prob = get_arrays(val_df)
    y_test, mlp_test_prob, vgam_test_prob = get_arrays(test_df)

    mlp_thresholds, _ = tune_model(y_val, mlp_val_prob)
    vgam_thresholds, _ = tune_model(y_val, vgam_val_prob)
    ensemble_config = tune_ensemble(
        y_val,
        mlp_val_prob,
        vgam_val_prob,
    )

    ensemble_val_prob = (
        ensemble_config["alpha_mlp"] * mlp_val_prob
        + ensemble_config["alpha_vgam"] * vgam_val_prob
    )
    ensemble_test_prob = (
        ensemble_config["alpha_mlp"] * mlp_test_prob
        + ensemble_config["alpha_vgam"] * vgam_test_prob
    )

    model_results = []

    for model_name, test_prob, thresholds in [
        ("MLP", mlp_test_prob, mlp_thresholds),
        ("VGAM", vgam_test_prob, vgam_thresholds),
        (
            "MLP-VGAM Ensemble",
            ensemble_test_prob,
            ensemble_config["thresholds"],
        ),
    ]:
        metrics, predictions = calculate_metrics(
            y_test,
            test_prob,
            thresholds,
        )

        model_results.append({
            "model": model_name,
            **metrics,
            **{
                f"threshold_{label}": float(thresholds[i])
                for i, label in enumerate(LABEL_COLS)
            },
        })

        report = classification_report(
            y_test,
            predictions,
            target_names=LABEL_COLS,
            zero_division=0,
        )
        with open(
            OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_report.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report)

    comparison = pd.DataFrame(model_results)
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    ensemble_pred = pd.DataFrame({ID_COL: test_df[ID_COL].astype(str)})
    for i, label in enumerate(LABEL_COLS):
        ensemble_pred[f"true_{label}"] = y_test[:, i]
        ensemble_pred[f"mlp_prob_{label}"] = mlp_test_prob[:, i]
        ensemble_pred[f"vgam_prob_{label}"] = vgam_test_prob[:, i]
        ensemble_pred[f"ensemble_prob_{label}"] = ensemble_test_prob[:, i]
        ensemble_pred[f"ensemble_pred_{label}"] = (
            ensemble_test_prob[:, i]
            >= ensemble_config["thresholds"][i]
        ).astype(int)

    ensemble_pred.to_csv(
        OUTPUT_DIR / "ensemble_test_predictions.csv",
        index=False,
    )

    config_to_save = {
        "alpha_mlp": ensemble_config["alpha_mlp"],
        "alpha_vgam": ensemble_config["alpha_vgam"],
        "validation_macro_f1": ensemble_config["validation_macro_f1"],
        "thresholds": {
            label: float(ensemble_config["thresholds"][i])
            for i, label in enumerate(LABEL_COLS)
        },
    }

    with open(
        OUTPUT_DIR / "ensemble_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config_to_save, file, indent=2, ensure_ascii=False)

    print("\nBobot ensemble terbaik:")
    print(json.dumps(config_to_save, indent=2))
    print("\nPerbandingan test:")
    print(comparison.to_string(index=False))
    print(f"\nHasil: {OUTPUT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
