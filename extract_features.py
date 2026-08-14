import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from train_effnet import (
    CoffeeLeafDataset,
    get_eval_transform,
    load_config,
    set_seed,
)


class EfficientNetB0FeatureExtractor(nn.Module):
    """Mengambil vektor setelah Global Average Pooling, sebelum classifier."""

    def __init__(self, trained_model: nn.Module):
        super().__init__()
        self.features = trained_model.features
        self.avgpool = trained_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def build_checkpoint_model(num_labels: int) -> nn.Module:
    """
    Membuat arsitektur yang sama dengan train_effnet.py tanpa mengunduh
    bobot ImageNet, karena seluruh bobot akan dimuat dari best_model.pt.
    """
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_labels)
    return model


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {path}\n"
            "Jalankan train_effnet.py terlebih dahulu."
        )

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def remove_module_prefix(state_dict: dict) -> dict:
    """Mendukung checkpoint yang pernah disimpan melalui DataParallel."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def rebuild_splits_if_needed(cfg: dict, output_dir: Path) -> None:
    """
    Fallback jika CSV split belum tersedia. Logikanya dibuat sama dengan
    train_effnet.py agar pembagian data tetap konsisten.
    """
    train_split = output_dir / "train_split.csv"
    val_split = output_dir / "val_split.csv"
    test_split = output_dir / "test_split.csv"

    if train_split.exists() and val_split.exists() and test_split.exists():
        return

    print("CSV split belum lengkap. Membuat ulang split sesuai config.yaml...")

    label_cols = cfg["label_cols"]
    train_full = pd.read_csv(cfg["train_csv"])
    test_df = pd.read_csv(cfg["test_csv"])

    for col in label_cols:
        train_full[col] = train_full[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    combo = train_full[label_cols].astype(str).agg("_".join, axis=1)
    stratify = combo if combo.value_counts().min() >= 2 else None

    train_df, val_df = train_test_split(
        train_full,
        test_size=cfg["val_size"],
        random_state=cfg["random_state"],
        stratify=stratify,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_split, index=False)
    val_df.to_csv(val_split, index=False)
    test_df.to_csv(test_split, index=False)


@torch.inference_mode()
def extract_split(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    extractor.eval()

    features_all = []
    labels_all = []
    ids_all = []

    progress = tqdm(loader, desc=f"Extract {split_name}")

    for images, labels, image_ids in progress:
        images = images.to(device, non_blocking=True)
        features = extractor(images)

        features_all.append(features.cpu().numpy().astype(np.float32))
        labels_all.append(labels.numpy().astype(np.float32))
        ids_all.extend(image_ids)

        progress.set_postfix(feature_dim=features.shape[1])

    if not features_all:
        raise ValueError(f"Split {split_name} tidak memiliki data.")

    return (
        np.concatenate(features_all, axis=0),
        np.concatenate(labels_all, axis=0),
        ids_all,
    )


def save_split_outputs(
    feature_dir: Path,
    split_name: str,
    id_col: str,
    label_cols: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    image_ids: list[str],
    save_csv: bool,
) -> pd.DataFrame:
    feature_dir.mkdir(parents=True, exist_ok=True)

    np.save(feature_dir / f"{split_name}_features.npy", features)
    np.save(feature_dir / f"{split_name}_labels.npy", labels)

    metadata = pd.DataFrame({
        id_col: image_ids,
        "split": split_name,
    })

    for index, label in enumerate(label_cols):
        metadata[label] = labels[:, index].astype(int)

    metadata.to_csv(feature_dir / f"{split_name}_metadata.csv", index=False)

    feature_columns = [
        f"feature_{index:04d}" for index in range(1, features.shape[1] + 1)
    ]
    feature_df = pd.DataFrame(features, columns=feature_columns)
    combined_df = pd.concat(
        [metadata.reset_index(drop=True), feature_df],
        axis=1,
    )

    if save_csv:
        combined_df.to_csv(
            feature_dir / f"{split_name}_feature_vectors.csv",
            index=False,
        )

    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ekstraksi fitur 1D EfficientNet-B0 untuk data kopi multi-label."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Default: <output_dir>/best_model.pt",
    )
    parser.add_argument(
        "--feature-dir",
        default=None,
        help="Default: <output_dir>/features",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Default: batch_size pada config.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Hanya simpan NPY dan metadata; tidak menyimpan matriks fitur CSV.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["random_state"])

    output_dir = Path(cfg["output_dir"])
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_dir / "best_model.pt"
    )
    feature_dir = (
        Path(args.feature_dir)
        if args.feature_dir
        else output_dir / "features"
    )
    batch_size = args.batch_size or cfg["batch_size"]

    rebuild_splits_if_needed(cfg, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output    : {feature_dir}")

    checkpoint = load_checkpoint(checkpoint_path, device)

    label_cols = checkpoint.get("label_cols", cfg["label_cols"])
    img_size = int(checkpoint.get("img_size", cfg["img_size"]))
    id_col = cfg["id_col"]

    model = build_checkpoint_model(num_labels=len(label_cols))
    state_dict = remove_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    extractor = EfficientNetB0FeatureExtractor(model).to(device)
    extractor.eval()

    split_specs = {
        "train": {
            "csv": output_dir / "train_split.csv",
            "image_dir": cfg["train_image_dir"],
        },
        "val": {
            "csv": output_dir / "val_split.csv",
            "image_dir": cfg["train_image_dir"],
        },
        "test": {
            "csv": output_dir / "test_split.csv",
            "image_dir": cfg["test_image_dir"],
        },
    }

    all_frames = []
    extraction_summary = {
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint.get("model_name", "efficientnet_b0"),
        "feature_layer": "avgpool_before_classifier",
        "feature_dimension": None,
        "img_size": img_size,
        "label_cols": label_cols,
        "splits": {},
    }

    for split_name in args.splits:
        spec = split_specs[split_name]
        split_df = pd.read_csv(spec["csv"])

        for col in label_cols:
            split_df[col] = split_df[col].astype(int)

        dataset = CoffeeLeafDataset(
            df=split_df,
            image_dir=spec["image_dir"],
            id_col=id_col,
            label_cols=label_cols,
            transform=get_eval_transform(img_size),
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        features, labels, image_ids = extract_split(
            extractor=extractor,
            loader=loader,
            device=device,
            split_name=split_name,
        )

        combined_df = save_split_outputs(
            feature_dir=feature_dir,
            split_name=split_name,
            id_col=id_col,
            label_cols=label_cols,
            features=features,
            labels=labels,
            image_ids=image_ids,
            save_csv=not args.skip_csv,
        )
        all_frames.append(combined_df)

        extraction_summary["feature_dimension"] = int(features.shape[1])
        extraction_summary["splits"][split_name] = {
            "n_samples": int(features.shape[0]),
            "feature_shape": list(features.shape),
            "label_shape": list(labels.shape),
        }

        print(f"{split_name:>5}: X={features.shape}, Y={labels.shape}")

    if not args.skip_csv:
        pd.concat(all_frames, ignore_index=True).to_csv(
            feature_dir / "all_feature_vectors.csv",
            index=False,
        )

    with open(
        feature_dir / "extraction_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(extraction_summary, file, indent=2, ensure_ascii=False)

    print("\nEkstraksi selesai.")
    print(f"Dimensi fitur per gambar: {extraction_summary['feature_dimension']}")
    print(f"Hasil tersimpan di      : {feature_dir}")


if __name__ == "__main__":
    main()
