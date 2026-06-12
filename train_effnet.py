import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import f1_score, hamming_loss, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(image_dir, image_id):
    """
    Mencari file gambar berdasarkan kolom id.
    Mendukung format:
    0.jpg, 0.jpeg, 0.png, 0.JPG, dst.
    """
    image_dir = Path(image_dir)
    sid = str(image_id)

    # Jika id terbaca sebagai 0.0, ubah ke 0
    try:
        if float(sid).is_integer():
            sid = str(int(float(sid)))
    except Exception:
        pass

    candidates = [
        sid,
        f"{sid}.jpg",
        f"{sid}.jpeg",
        f"{sid}.png",
        f"{sid}.JPG",
        f"{sid}.JPEG",
        f"{sid}.PNG",
    ]

    for name in candidates:
        path = image_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Gambar untuk id={image_id} tidak ditemukan di {image_dir}. "
        f"Contoh kandidat: {candidates}"
    )


class CoffeeLeafDataset(Dataset):
    def __init__(self, df, image_dir, id_col, label_cols, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.id_col = id_col
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row[self.id_col]
        img_path = resolve_image_path(self.image_dir, image_id)

        image = Image.open(img_path).convert("RGB")
        labels = row[self.label_cols].astype(float).values
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, str(image_id)


def get_train_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_model(num_labels):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_labels)

    return model


def compute_pos_weight(df, label_cols):
    y = df[label_cols].values.astype(float)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    pos_weight = neg / np.clip(pos, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs_all = []
    targets_all = []
    ids_all = []

    for images, labels, ids in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        probs_all.append(probs)
        targets_all.append(labels.numpy())
        ids_all.extend(ids)

    return np.vstack(probs_all), np.vstack(targets_all), ids_all


def find_best_thresholds(y_true, y_prob, label_cols):
    thresholds = {}

    for i, label in enumerate(label_cols):
        best_t = 0.5
        best_f1 = -1

        for t in np.arange(0.10, 0.91, 0.01):
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))

        thresholds[label] = best_t

    return thresholds


def evaluate_metrics(y_true, y_prob, thresholds, label_cols):
    th = np.array([thresholds[label] for label in label_cols])
    y_pred = (y_prob >= th).astype(int)

    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    try:
        metrics["mAP_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
        metrics["mAP_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
    except Exception:
        metrics["mAP_macro"] = None
        metrics["mAP_micro"] = None

    metrics["per_label"] = {}
    for i, label in enumerate(label_cols):
        pred_i = y_pred[:, i]
        true_i = y_true[:, i]

        metrics["per_label"][label] = {
            "threshold": thresholds[label],
            "f1": float(f1_score(true_i, pred_i, zero_division=0)),
            "positive_count": int(true_i.sum())
        }

    return metrics, y_pred


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["random_state"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    label_cols = cfg["label_cols"]
    id_col = cfg["id_col"]

    train_full = pd.read_csv(cfg["train_csv"])
    test_df = pd.read_csv(cfg["test_csv"])

    for col in label_cols:
        train_full[col] = train_full[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    # Stratifikasi sederhana berdasarkan kombinasi multi-label
    combo = train_full[label_cols].astype(str).agg("_".join, axis=1)
    stratify = combo if combo.value_counts().min() >= 2 else None

    train_df, val_df = train_test_split(
        train_full,
        test_size=cfg["val_size"],
        random_state=cfg["random_state"],
        stratify=stratify
    )

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    print("Jumlah data:")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")
    print()
    print("Distribusi label train:")
    print(train_df[label_cols].sum())

    train_ds = CoffeeLeafDataset(
        train_df,
        cfg["train_image_dir"],
        id_col,
        label_cols,
        transform=get_train_transform(cfg["img_size"])
    )

    val_ds = CoffeeLeafDataset(
        val_df,
        cfg["train_image_dir"],
        id_col,
        label_cols,
        transform=get_eval_transform(cfg["img_size"])
    )

    test_ds = CoffeeLeafDataset(
        test_df,
        cfg["test_image_dir"],
        id_col,
        label_cols,
        transform=get_eval_transform(cfg["img_size"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = build_model(num_labels=len(label_cols)).to(device)

    if cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weight(train_df, label_cols).to(device)
        print("pos_weight:", pos_weight.detach().cpu().numpy())
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_macro_f1 = -1
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")

        for images, labels, _ in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            loop.set_postfix(loss=np.mean(losses))

        val_prob, val_true, _ = predict_loader(model, val_loader, device)
        thresholds = find_best_thresholds(val_true, val_prob, label_cols)
        val_metrics, _ = evaluate_metrics(val_true, val_prob, thresholds, label_cols)

        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = float(np.mean(losses))
        history.append(val_metrics)

        print(
            f"Epoch {epoch}: "
            f"loss={np.mean(losses):.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"micro_f1={val_metrics['micro_f1']:.4f} | "
            f"hamming={val_metrics['hamming_loss']:.4f}"
        )
        print("Thresholds:", thresholds)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]

            torch.save({
                "model_state_dict": model.state_dict(),
                "label_cols": label_cols,
                "thresholds": thresholds,
                "img_size": cfg["img_size"],
                "model_name": "efficientnet_b0",
                "config": cfg
            }, output_dir / "best_model.pt")

            save_json(thresholds, output_dir / "thresholds.json")
            save_json(val_metrics, output_dir / "best_val_metrics.json")

            print("Best model updated.")

    save_json(history, output_dir / "training_history.json")

    print()
    print("Evaluasi test set dengan best model...")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_prob, test_true, test_ids = predict_loader(model, test_loader, device)
    thresholds = checkpoint["thresholds"]
    test_metrics, test_pred = evaluate_metrics(test_true, test_prob, thresholds, label_cols)

    print()
    print("TEST METRICS:")
    print(json.dumps(test_metrics, indent=2))

    print()
    print("CLASSIFICATION REPORT:")
    print(classification_report(test_true, test_pred, target_names=label_cols, zero_division=0))

    pred_df = pd.DataFrame({
        "id": test_ids
    })

    for i, label in enumerate(label_cols):
        pred_df[f"true_{label}"] = test_true[:, i].astype(int)
        pred_df[f"prob_{label}"] = test_prob[:, i]
        pred_df[f"pred_{label}"] = test_pred[:, i].astype(int)

    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json(test_metrics, output_dir / "test_metrics.json")

    print()
    print("Selesai.")
    print(f"Model terbaik: {output_dir / 'best_model.pt'}")
    print(f"Metrik test : {output_dir / 'test_metrics.json'}")
    print(f"Prediksi    : {output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()