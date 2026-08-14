"""
01_prepare_train_mlp.py

Pipeline:
1. Membaca deep features hasil EfficientNet.
2. Membagi train menjadi training dan validation.
3. StandardScaler + PCA (fit hanya pada training).
4. Menyimpan data PCA agar dipakai oleh VGAM di R.
5. Melatih MLP multi-label.
6. Menyimpan probabilitas validation dan test.

Jalankan:
    python 01_prepare_train_mlp.py
"""

import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# KONFIGURASI
# ============================================================
TRAIN_FEATURES_CSV = Path("outputs/tabular_features/train_features.csv")
TEST_FEATURES_CSV = Path("outputs/tabular_features/test_features.csv")
OUTPUT_DIR = Path("outputs/modeling")

ID_COL = "id"
LABEL_COLS = ["miner", "rust", "phoma"]

RANDOM_STATE = 42
VAL_SIZE = 0.20
PCA_COMPONENTS = 12

HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.25
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 250
PATIENCE = 25


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan extract_features_simple.py terlebih dahulu."
        )

    df = pd.read_csv(path, dtype={ID_COL: str})

    missing_labels = [col for col in LABEL_COLS if col not in df.columns]
    if missing_labels:
        raise ValueError(f"Kolom label tidak ditemukan: {missing_labels}")

    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    if not feature_cols:
        raise ValueError("Tidak ditemukan kolom feature_1, feature_2, dan seterusnya.")

    numeric_cols = feature_cols + LABEL_COLS
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    if df[numeric_cols].isna().any().any():
        bad_cols = df[numeric_cols].columns[df[numeric_cols].isna().any()].tolist()
        raise ValueError(f"Terdapat missing/non-numeric value pada: {bad_cols[:10]}")

    df[LABEL_COLS] = df[LABEL_COLS].astype(int)
    return df


def split_multilabel(df: pd.DataFrame):
    """Stratifikasi berdasarkan kombinasi label jika memungkinkan."""
    label_combo = df[LABEL_COLS].astype(str).agg("_".join, axis=1)
    stratify = label_combo if label_combo.value_counts().min() >= 2 else None

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_prepared_dataframe(
    source_df: pd.DataFrame,
    x_pca: np.ndarray,
    pc_cols: list[str],
    split_name: str,
) -> pd.DataFrame:
    metadata = source_df[[ID_COL] + LABEL_COLS].reset_index(drop=True).copy()
    metadata["split"] = split_name
    pc_df = pd.DataFrame(x_pca, columns=pc_cols)
    return pd.concat([metadata, pc_df], axis=1)


class MultilabelMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.LayerNorm(HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    probabilities = []

    for (features,) in loader:
        logits = model(features.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.vstack(probabilities)


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    thresholds = []

    for label_index in range(y_true.shape[1]):
        best_threshold = 0.50
        best_f1 = -1.0

        for threshold in np.arange(0.10, 0.91, 0.01):
            prediction = (y_prob[:, label_index] >= threshold).astype(int)
            score = f1_score(
                y_true[:, label_index],
                prediction,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(threshold, 2))

        thresholds.append(best_threshold)

    return np.asarray(thresholds, dtype=float)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
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

    return metrics


def make_prediction_dataframe(
    ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame({ID_COL: ids.astype(str).values})

    for i, label in enumerate(LABEL_COLS):
        result[f"true_{label}"] = y_true[:, i].astype(int)
        result[f"prob_{label}"] = y_prob[:, i]

    return result


def main() -> None:
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_dir = OUTPUT_DIR / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    print("Membaca deep features...")
    train_full = load_feature_data(TRAIN_FEATURES_CSV)
    test_df = load_feature_data(TEST_FEATURES_CSV)

    feature_cols = [col for col in train_full.columns if col.startswith("feature_")]
    train_df, val_df = split_multilabel(train_full)

    x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    x_val_raw = val_df[feature_cols].to_numpy(dtype=np.float32)
    x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)

    y_train = train_df[LABEL_COLS].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_COLS].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COLS].to_numpy(dtype=np.float32)

    print(f"Train: {x_train_raw.shape}")
    print(f"Val  : {x_val_raw.shape}")
    print(f"Test : {x_test_raw.shape}")

    # Fit preprocessing hanya pada training set.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    max_components = min(
        PCA_COMPONENTS,
        x_train_scaled.shape[0] - 1,
        x_train_scaled.shape[1],
    )
    if max_components < 2:
        raise ValueError("Data training terlalu sedikit untuk PCA.")

    pca = PCA(n_components=max_components, random_state=RANDOM_STATE)
    x_train = pca.fit_transform(x_train_scaled).astype(np.float32)
    x_val = pca.transform(x_val_scaled).astype(np.float32)
    x_test = pca.transform(x_test_scaled).astype(np.float32)

    pc_cols = [f"PC{i:02d}" for i in range(1, max_components + 1)]

    make_prepared_dataframe(
        train_df, x_train, pc_cols, "train"
    ).to_csv(prepared_dir / "train_pca.csv", index=False)

    make_prepared_dataframe(
        val_df, x_val, pc_cols, "val"
    ).to_csv(prepared_dir / "val_pca.csv", index=False)

    make_prepared_dataframe(
        test_df, x_test, pc_cols, "test"
    ).to_csv(prepared_dir / "test_pca.csv", index=False)

    joblib.dump(scaler, OUTPUT_DIR / "standard_scaler.joblib")
    joblib.dump(pca, OUTPUT_DIR / "pca.joblib")

    print(
        f"PCA: {max_components} komponen, "
        f"explained variance={pca.explained_variance_ratio_.sum():.4f}"
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = MultilabelMLP(
        input_dim=x_train.shape[1],
        output_dim=len(LABEL_COLS),
    ).to(device)

    positive = y_train.sum(axis=0)
    negative = len(y_train) - positive
    pos_weight = negative / np.clip(positive, 1.0, None)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        val_prob = predict_probabilities(model, x_val, device)
        val_pred_05 = (val_prob >= 0.50).astype(int)
        val_macro_f1 = f1_score(
            y_val,
            val_pred_05,
            average="macro",
            zero_division=0,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_macro_f1_at_0.5": float(val_macro_f1),
        })

        if val_macro_f1 > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={np.mean(epoch_losses):.4f} | "
                f"val macro-F1={val_macro_f1:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping pada epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Model MLP gagal memperoleh state terbaik.")

    model.load_state_dict(best_state)
    model.eval()

    val_prob = predict_probabilities(model, x_val, device)
    test_prob = predict_probabilities(model, x_test, device)

    thresholds = find_best_thresholds(y_val, val_prob)
    val_metrics = calculate_metrics(y_val, val_prob, thresholds)
    test_metrics = calculate_metrics(y_test, test_prob, thresholds)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "label_cols": LABEL_COLS,
            "thresholds": thresholds.tolist(),
            "config": {
                "pca_components": max_components,
                "hidden_1": HIDDEN_1,
                "hidden_2": HIDDEN_2,
                "dropout": DROPOUT,
            },
        },
        OUTPUT_DIR / "mlp_model.pt",
    )

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "mlp_training_history.csv",
        index=False,
    )

    make_prediction_dataframe(
        val_df[ID_COL], y_val, val_prob
    ).to_csv(OUTPUT_DIR / "mlp_val_predictions.csv", index=False)

    make_prediction_dataframe(
        test_df[ID_COL], y_test, test_prob
    ).to_csv(OUTPUT_DIR / "mlp_test_predictions.csv", index=False)

    result = {
        "label_cols": LABEL_COLS,
        "thresholds": {
            label: float(thresholds[i])
            for i, label in enumerate(LABEL_COLS)
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pca_components": max_components,
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
    }

    with open(OUTPUT_DIR / "mlp_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\nMLP selesai.")
    print("Threshold:", result["thresholds"])
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
