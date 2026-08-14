# Coffee Multi-label Modeling Pipeline

Pipeline ini membandingkan:

1. MLP multi-label
2. Vector Generalized Additive Model (VGAM)
3. Weighted soft-voting ensemble MLP–VGAM

## Input yang diperlukan

Pastikan dua file berikut sudah tersedia:

```text
outputs/tabular_features/train_features.csv
outputs/tabular_features/test_features.csv
```

Struktur minimum:

```text
id, miner, rust, phoma, feature_1, ..., feature_1280
```

## Urutan menjalankan

### 1. Aktifkan environment Python

```bash
conda activate kopi-demo
```

Instal dependensi tambahan jika perlu:

```bash
pip install -r requirements_modeling.txt
```

### 2. Preprocessing dan training MLP

```bash
python 01_prepare_train_mlp.py
```

Script ini membuat split training/validation, melakukan StandardScaler dan PCA,
melatih MLP, lalu menyimpan data PCA yang sama untuk VGAM.

### 3. Training VGAM dengan R

Aktifkan environment R:

```bash
conda activate r-kopi
```

Jalankan:

```bash
Rscript 02_train_vgam.R
```

Atau buka `02_train_vgam.R` di VS Code dan jalankan menggunakan kernel R.

### 4. Evaluasi ensemble

Kembali ke environment Python:

```bash
conda activate kopi-demo
python 03_evaluate_ensemble.py
```

## Output utama

```text
outputs/modeling/
├── prepared/
│   ├── train_pca.csv
│   ├── val_pca.csv
│   └── test_pca.csv
├── mlp_model.pt
├── vgam_model.rds
├── mlp_test_predictions.csv
├── vgam_test_predictions.csv
├── ensemble_test_predictions.csv
├── ensemble_config.json
└── model_comparison.csv
```

`model_comparison.csv` berisi metrik MLP, VGAM, dan ensemble:

- micro-F1
- macro-F1
- samples-F1
- Hamming loss
- subset accuracy
- macro-AUPRC
- micro-AUPRC

## Parameter yang mudah diubah

Pada `01_prepare_train_mlp.py`:

```python
PCA_COMPONENTS = 12
HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.25
```

Pada `02_train_vgam.R`:

```r
SMOOTH_DF <- 3
```

Untuk eksperimen paper, ubah satu parameter pada satu waktu dan simpan hasil setiap
skenario secara terpisah.
