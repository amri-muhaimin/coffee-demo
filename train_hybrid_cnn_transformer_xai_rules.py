import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ============================================================
# SISTEM DAN METODE DETEKSI MULTI-LABEL PENYAKIT DAUN KOPI
# HYBRID CNN-TRANSFORMER + EXPLAINABLE AI + REKOMENDASI ATURAN
# ============================================================
#
# Ringkasan arsitektur:
# Citra daun kopi
#   -> EfficientNet-B0 sebagai CNN feature extractor
#   -> proyeksi feature map CNN menjadi embedding token
#   -> Transformer Encoder untuk hubungan spasial/kontekstual
#   -> classifier multi-label dengan BCEWithLogitsLoss
#   -> Grad-CAM sebagai Explainable AI
#   -> rekomendasi pengendalian berbasis aturan
#
# Cara menjalankan:
#   python train_hybrid_cnn_transformer_xai_rules.py --config config_hybrid.yaml
#
# Catatan:
# - Script ini mempertahankan struktur config/data dari train_effnet.py.
# - Label default mengikuti dataset Bapak: miner, rust, phoma.
# - Untuk GPU ringan seperti NVIDIA MX330, gunakan d_model=128,
#   transformer_layers=1 atau 2, batch_size=2 atau 4.


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_image_path(image_dir, image_id):
    """
    Mencari file gambar berdasarkan kolom id.
    Mendukung format: 0, 0.jpg, 0.jpeg, 0.png, 0.JPG, dst.
    """
    image_dir = Path(image_dir)
    sid = str(image_id)

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
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transform(img_size):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class HybridCNNTransformer(nn.Module):
    """
    EfficientNet-B0 sebagai CNN backbone, diikuti Transformer Encoder.

    Input:
        images: tensor [B, 3, H, W]

    Tahap:
        1. CNN menghasilkan feature map [B, C, h, w]
        2. 1x1 conv memproyeksikan C -> d_model
        3. feature map diratakan menjadi token spasial [B, h*w, d_model]
        4. CLS token + positional embedding
        5. Transformer Encoder memodelkan hubungan antar-token
        6. CLS output dipakai untuk multi-label classifier
    """

    def __init__(
        self,
        num_labels,
        d_model=128,
        nhead=4,
        transformer_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        pretrained=True,
        freeze_cnn=False,
        max_tokens=197,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        # CNN feature extractor tanpa classifier.
        self.cnn = effnet.features
        cnn_out_channels = 1280  # output channel EfficientNet-B0 features

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # Proyeksi feature map CNN menjadi dimensi token Transformer.
        self.proj = nn.Conv2d(cnn_out_channels, d_model, kernel_size=1)

        # CLS token dipakai seperti pada Vision Transformer.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional embedding dibuat cukup fleksibel.
        # Untuk img_size 224, token spasial EfficientNet-B0 umumnya 7x7=49.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        self.d_model = d_model
        self.max_tokens = max_tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward_features(self, x):
        """
        Menghasilkan:
        - feature_map: keluaran CNN [B, 1280, h, w]
        - tokens: token Transformer setelah proyeksi [B, h*w, d_model]
        """
        feature_map = self.cnn(x)
        z = self.proj(feature_map)              # [B, d_model, h, w]
        z = z.flatten(2).transpose(1, 2)        # [B, h*w, d_model]
        return feature_map, z

    def forward(self, x, return_attention_input=False):
        batch_size = x.shape[0]

        feature_map, tokens = self.forward_features(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)   # [B, 1+h*w, d_model]

        if seq.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(
                f"Jumlah token {seq.shape[1]} melebihi max_tokens={self.pos_embed.shape[1]}. "
                "Naikkan max_tokens di config."
            )

        seq = seq + self.pos_embed[:, : seq.shape[1], :]

        encoded = self.transformer(seq)
        cls_out = self.norm(encoded[:, 0])
        logits = self.classifier(cls_out)

        if return_attention_input:
            return logits, feature_map, tokens, encoded

        return logits


def build_model(num_labels, cfg):
    model = HybridCNNTransformer(
        num_labels=num_labels,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        transformer_layers=int(cfg.get("transformer_layers", 2)),
        dim_feedforward=int(cfg.get("dim_feedforward", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        pretrained=bool(cfg.get("pretrained", True)),
        freeze_cnn=bool(cfg.get("freeze_cnn", False)),
        max_tokens=int(cfg.get("max_tokens", 197)),
    )
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
            "positive_count": int(true_i.sum()),
        }

    return metrics, y_pred


RULE_BASED_RECOMMENDATIONS = {
    "miner": {
        "nama_penyakit": "Leaf miner / serangan pengorok daun",
        "rekomendasi": [
            "Pisahkan atau pangkas daun yang menunjukkan gejala berat untuk mengurangi sumber serangan.",
            "Lakukan sanitasi kebun dan pengamatan berkala pada daun muda.",
            "Gunakan perangkap atau pengendalian hayati bila tersedia dan sesuai kondisi lapangan.",
            "Konsultasikan penggunaan insektisida selektif kepada penyuluh pertanian setempat bila serangan meluas.",
        ],
    },
    "rust": {
        "nama_penyakit": "Karat daun kopi",
        "rekomendasi": [
            "Pangkas bagian tanaman yang terlalu rimbun agar sirkulasi udara lebih baik.",
            "Kurangi kelembapan berlebih melalui pengaturan naungan dan drainase.",
            "Kumpulkan daun terinfeksi yang gugur untuk mengurangi sumber inokulum.",
            "Pertimbangkan fungisida yang direkomendasikan penyuluh pertanian bila intensitas penyakit tinggi.",
        ],
    },
    "phoma": {
        "nama_penyakit": "Bercak daun Phoma",
        "rekomendasi": [
            "Lakukan sanitasi daun atau ranting yang menunjukkan bercak nekrotik berat.",
            "Hindari kelembapan berlebih dan perbaiki aerasi tajuk tanaman.",
            "Jaga nutrisi tanaman agar tanaman lebih toleran terhadap infeksi.",
            "Konsultasikan pengendalian fungisida sesuai rekomendasi teknis setempat bila penyakit menyebar.",
        ],
    },
}


def make_rule_based_recommendations(probs, label_cols, thresholds):
    """
    Menghasilkan rekomendasi berbasis aturan dari probabilitas multi-label.
    """
    results = []

    for i, label in enumerate(label_cols):
        prob = float(probs[i])
        th = float(thresholds.get(label, 0.5))

        if prob >= th:
            rule = RULE_BASED_RECOMMENDATIONS.get(
                label,
                {
                    "nama_penyakit": label,
                    "rekomendasi": [
                        "Lakukan validasi lapangan dan konsultasi dengan penyuluh pertanian.",
                    ],
                },
            )
            results.append(
                {
                    "label": label,
                    "nama_penyakit": rule["nama_penyakit"],
                    "probabilitas": round(prob, 4),
                    "threshold": th,
                    "rekomendasi": rule["rekomendasi"],
                }
            )

    if not results:
        results.append(
            {
                "label": "sehat/tidak_terdeteksi",
                "nama_penyakit": "Tidak ada label penyakit yang melewati ambang batas",
                "probabilitas": None,
                "threshold": None,
                "rekomendasi": [
                    "Lanjutkan pemantauan rutin kondisi daun kopi.",
                    "Ambil ulang citra dengan pencahayaan yang baik bila gejala visual masih diragukan.",
                ],
            }
        )

    return results


class GradCAM:
    """
    Grad-CAM pada feature map CNN terakhir sebelum Transformer.

    Heatmap menunjukkan area citra yang berkontribusi terhadap skor label.
    Karena modelnya hybrid, Grad-CAM ditempelkan pada keluaran CNN, lalu
    gradient dihitung terhadap skor output setelah Transformer.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = self.model.cnn.register_forward_hook(self._save_activation)
        self.backward_handle = self.model.cnn.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor, target_label_idx):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image_tensor)
        score = logits[:, target_label_idx].sum()
        score.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def save_gradcam_overlay(image_tensor, cam, out_path, alpha=0.35):
    import matplotlib.pyplot as plt

    img = denormalize_image(image_tensor.squeeze(0))

    plt.figure(figsize=(5, 5), dpi=160)
    plt.imshow(img)
    plt.imshow(cam, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def generate_xai_examples(model, loader, device, output_dir, label_cols, thresholds, max_samples=6):
    output_dir = Path(output_dir)
    xai_dir = output_dir / "xai_examples"
    xai_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model)
    model.eval()

    metadata = []
    count = 0

    for images, labels, ids in loader:
        for b in range(images.shape[0]):
            if count >= max_samples:
                gradcam.remove_hooks()
                save_json(metadata, xai_dir / "xai_metadata.json")
                return

            image = images[b : b + 1].to(device)

            with torch.no_grad():
                logits = model(image)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            th = np.array([thresholds.get(label, 0.5) for label in label_cols])
            positive_idx = np.where(probs >= th)[0]

            if len(positive_idx) > 0:
                target_idx = int(positive_idx[np.argmax(probs[positive_idx])])
            else:
                target_idx = int(np.argmax(probs))

            cam = gradcam(image, target_idx)
            target_label = label_cols[target_idx]

            out_file = xai_dir / f"gradcam_{ids[b]}_{target_label}.png"
            save_gradcam_overlay(image, cam, out_file)

            recs = make_rule_based_recommendations(probs, label_cols, thresholds)

            metadata.append(
                {
                    "id": ids[b],
                    "target_label_for_xai": target_label,
                    "probabilities": {
                        label: float(probs[i]) for i, label in enumerate(label_cols)
                    },
                    "recommendations": recs,
                    "gradcam_file": str(out_file),
                }
            )

            count += 1

    gradcam.remove_hooks()
    save_json(metadata, xai_dir / "xai_metadata.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_hybrid.yaml")
    parser.add_argument(
        "--explain-samples",
        type=int,
        default=None,
        help="Jumlah contoh Grad-CAM yang disimpan. Jika None, gunakan nilai config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("random_state", 42))

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    label_cols = cfg["label_cols"]
    id_col = cfg["id_col"]

    train_full = pd.read_csv(cfg["train_csv"])
    test_df = pd.read_csv(cfg["test_csv"])

    for col in label_cols:
        train_full[col] = train_full[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    combo = train_full[label_cols].astype(str).agg("_".join, axis=1)
    stratify = combo if combo.value_counts().min() >= 2 else None

    train_df, val_df = train_test_split(
        train_full,
        test_size=cfg.get("val_size", 0.15),
        random_state=cfg.get("random_state", 42),
        stratify=stratify,
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
        transform=get_train_transform(cfg.get("img_size", 224)),
    )
    val_ds = CoffeeLeafDataset(
        val_df,
        cfg["train_image_dir"],
        id_col,
        label_cols,
        transform=get_eval_transform(cfg.get("img_size", 224)),
    )
    test_ds = CoffeeLeafDataset(
        test_df,
        cfg["test_image_dir"],
        id_col,
        label_cols,
        transform=get_eval_transform(cfg.get("img_size", 224)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print()
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = build_model(num_labels=len(label_cols), cfg=cfg).to(device)
    print()
    print("Model:", cfg.get("model_name", "hybrid_effnet_b0_transformer"))
    print(
        "Transformer:",
        {
            "d_model": cfg.get("d_model", 128),
            "nhead": cfg.get("nhead", 4),
            "transformer_layers": cfg.get("transformer_layers", 2),
            "dim_feedforward": cfg.get("dim_feedforward", 256),
        },
    )

    if cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weight(train_df, label_cols).to(device)
        print("pos_weight:", pos_weight.detach().cpu().numpy())
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 0.0003),
        weight_decay=cfg.get("weight_decay", 0.0001),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_macro_f1 = -1
    history = []

    for epoch in range(1, cfg.get("epochs", 10) + 1):
        model.train()
        losses = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.get('epochs', 10)}")
        for images, labels, _ in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if cfg.get("grad_clip", 1.0) is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("grad_clip", 1.0)))

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

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_cols": label_cols,
                    "thresholds": thresholds,
                    "img_size": cfg.get("img_size", 224),
                    "model_name": cfg.get("model_name", "hybrid_effnet_b0_transformer"),
                    "architecture": "EfficientNet-B0 CNN + Transformer Encoder + Multi-label Classifier",
                    "rule_base": RULE_BASED_RECOMMENDATIONS,
                    "config": cfg,
                },
                output_dir / "best_hybrid_model.pt",
            )

            save_json(thresholds, output_dir / "thresholds.json")
            save_json(val_metrics, output_dir / "best_val_metrics.json")
            print("Best hybrid model updated.")

    save_json(history, output_dir / "training_history.json")

    print()
    print("Evaluasi test set dengan best hybrid model...")

    checkpoint = torch.load(output_dir / "best_hybrid_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_prob, test_true, test_ids = predict_loader(model, test_loader, device)
    thresholds = checkpoint["thresholds"]
    test_metrics, test_pred = evaluate_metrics(test_true, test_prob, thresholds, label_cols)

    print()
    print("TEST METRICS:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    print()
    print("CLASSIFICATION REPORT:")
    print(classification_report(test_true, test_pred, target_names=label_cols, zero_division=0))

    pred_df = pd.DataFrame({"id": test_ids})
    for i, label in enumerate(label_cols):
        pred_df[f"true_{label}"] = test_true[:, i].astype(int)
        pred_df[f"prob_{label}"] = test_prob[:, i]
        pred_df[f"pred_{label}"] = test_pred[:, i].astype(int)

    recommendation_records = []
    for row_idx, image_id in enumerate(test_ids):
        probs = test_prob[row_idx]
        recs = make_rule_based_recommendations(probs, label_cols, thresholds)
        recommendation_records.append({"id": image_id, "recommendations": recs})

    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json(test_metrics, output_dir / "test_metrics.json")
    save_json(recommendation_records, output_dir / "test_recommendations.json")

    explain_samples = (
        args.explain_samples
        if args.explain_samples is not None
        else int(cfg.get("explain_samples", 6))
    )

    if explain_samples > 0:
        print()
        print(f"Membuat {explain_samples} contoh Grad-CAM XAI...")
        generate_xai_examples(
            model=model,
            loader=test_loader,
            device=device,
            output_dir=output_dir,
            label_cols=label_cols,
            thresholds=thresholds,
            max_samples=explain_samples,
        )

    print()
    print("Selesai.")
    print(f"Model terbaik     : {output_dir / 'best_hybrid_model.pt'}")
    print(f"Metrik test       : {output_dir / 'test_metrics.json'}")
    print(f"Prediksi          : {output_dir / 'test_predictions.csv'}")
    print(f"Rekomendasi aturan: {output_dir / 'test_recommendations.json'}")
    if explain_samples > 0:
        print(f"Contoh XAI        : {output_dir / 'xai_examples'}")


if __name__ == "__main__":
    main()
