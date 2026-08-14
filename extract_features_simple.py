from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# =========================
# PENGATURAN
# =========================
TRAIN_CSV = "train_classes.csv"
TEST_CSV = "test_classes.csv"

TRAIN_IMAGE_DIR = "coffee-leaf-diseases/train/images"
TEST_IMAGE_DIR = "coffee-leaf-diseases/test/images"

ID_COL = "id"
LABEL_COLS = ["miner", "rust", "phoma"]

IMG_SIZE = 224
BATCH_SIZE = 8
OUTPUT_DIR = Path("outputs/tabular_features")


def find_image(image_dir, image_id):
    image_dir = Path(image_dir)

    try:
        image_id = str(int(float(image_id)))
    except Exception:
        image_id = str(image_id)

    candidates = [
        image_dir / image_id,
        image_dir / f"{image_id}.jpg",
        image_dir / f"{image_id}.jpeg",
        image_dir / f"{image_id}.png",
        image_dir / f"{image_id}.JPG",
        image_dir / f"{image_id}.JPEG",
        image_dir / f"{image_id}.PNG",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Gambar id={image_id} tidak ditemukan.")


class CoffeeDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = find_image(
            self.image_dir,
            row[ID_COL],
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = row[LABEL_COLS].astype(int).values

        return image, str(row[ID_COL]), torch.tensor(labels)


def build_feature_extractor(device):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        torch.nn.Flatten(start_dim=1),
    )

    extractor = extractor.to(device)
    extractor.eval()

    return extractor


@torch.no_grad()
def extract_to_csv(csv_path, image_dir, output_csv, extractor, device):
    dataset = CoffeeDataset(csv_path, image_dir)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    rows = []

    for images, image_ids, labels in tqdm(loader, desc=f"Extract {output_csv.name}"):
        images = images.to(device)

        features = extractor(images).cpu()

        for i in range(len(image_ids)):
            row = {
                ID_COL: image_ids[i],
                "miner": int(labels[i][0]),
                "rust": int(labels[i][1]),
                "phoma": int(labels[i][2]),
            }

            for j, value in enumerate(features[i].tolist(), start=1):
                row[f"feature_{j}"] = value

            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_csv, index=False)

    print(f"Tersimpan: {output_csv}")
    print(f"Ukuran data: {result.shape}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    extractor = build_feature_extractor(device)

    extract_to_csv(
        TRAIN_CSV,
        TRAIN_IMAGE_DIR,
        OUTPUT_DIR / "train_features.csv",
        extractor,
        device,
    )

    extract_to_csv(
        TEST_CSV,
        TEST_IMAGE_DIR,
        OUTPUT_DIR / "test_features.csv",
        extractor,
        device,
    )


if __name__ == "__main__":
    main()
