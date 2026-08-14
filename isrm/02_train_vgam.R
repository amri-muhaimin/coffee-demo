# 02_train_vgam.R
#
# Melatih Vector Generalized Additive Model (VGAM) multi-label
# menggunakan data PCA yang dibuat oleh 01_prepare_train_mlp.py.
#
# Jalankan dari terminal:
#   Rscript 02_train_vgam.R
#
# Atau buka file ini di VS Code dengan kernel R dan jalankan per bagian.

options(stringsAsFactors = FALSE)

OUTPUT_DIR <- "outputs/modeling"
PREPARED_DIR <- file.path(OUTPUT_DIR, "prepared")

TRAIN_CSV <- file.path(PREPARED_DIR, "train_pca.csv")
VAL_CSV <- file.path(PREPARED_DIR, "val_pca.csv")
TEST_CSV <- file.path(PREPARED_DIR, "test_pca.csv")

LABEL_COLS <- c("miner", "rust", "phoma")
SMOOTH_DF <- 3

if (!requireNamespace("VGAM", quietly = TRUE)) {
  stop(
    paste(
      "Package VGAM belum terpasang.",
      "Pasang dengan: install.packages('VGAM')",
      "atau: conda install -n r-kopi -c conda-forge r-vgam",
      sep = "\n"
    )
  )
}

library(VGAM)

required_files <- c(TRAIN_CSV, VAL_CSV, TEST_CSV)
missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    paste(
      "Data PCA belum tersedia. Jalankan 01_prepare_train_mlp.py dahulu.",
      paste(missing_files, collapse = "\n"),
      sep = "\n"
    )
  )
}

train_data <- read.csv(TRAIN_CSV, check.names = FALSE)
val_data <- read.csv(VAL_CSV, check.names = FALSE)
test_data <- read.csv(TEST_CSV, check.names = FALSE)

pc_cols <- grep("^PC[0-9]+$", names(train_data), value = TRUE)

if (length(pc_cols) == 0) {
  stop("Kolom PCA tidak ditemukan.")
}

for (label in LABEL_COLS) {
  train_data[[label]] <- as.integer(train_data[[label]])
  val_data[[label]] <- as.integer(val_data[[label]])
  test_data[[label]] <- as.integer(test_data[[label]])
}

# Formula:
# cbind(miner, rust, phoma) ~ s(PC01, df=3) + ... + s(PC12, df=3)
response_part <- paste0(
  "cbind(",
  paste(LABEL_COLS, collapse = ", "),
  ")"
)

smooth_terms <- paste0(
  "s(",
  pc_cols,
  ", df = ",
  SMOOTH_DF,
  ")"
)

model_formula <- as.formula(
  paste(
    response_part,
    "~",
    paste(smooth_terms, collapse = " + ")
  )
)

cat("Formula VGAM:\n")
print(model_formula)
cat("\nTraining VGAM...\n")

vgam_model <- vgam(
  formula = model_formula,
  family = binomialff(
    link = "logitlink",
    multiple.responses = TRUE,
    parallel = FALSE
  ),
  data = train_data,
  control = vgam.control(
    maxit = 100,
    trace = TRUE
  ),
  model = TRUE
)

saveRDS(vgam_model, file.path(OUTPUT_DIR, "vgam_model.rds"))

val_prob <- as.matrix(
  predict(vgam_model, newdata = val_data, type = "response")
)
test_prob <- as.matrix(
  predict(vgam_model, newdata = test_data, type = "response")
)

if (ncol(val_prob) != length(LABEL_COLS)) {
  stop(
    paste0(
      "Jumlah output probabilitas VGAM tidak sesuai. Diperoleh ",
      ncol(val_prob),
      ", diharapkan ",
      length(LABEL_COLS),
      "."
    )
  )
}

colnames(val_prob) <- LABEL_COLS
colnames(test_prob) <- LABEL_COLS

# Pastikan probabilitas tetap dalam rentang numerik aman.
val_prob <- pmin(pmax(val_prob, 1e-7), 1 - 1e-7)
test_prob <- pmin(pmax(test_prob, 1e-7), 1 - 1e-7)

make_prediction_data <- function(source_data, probabilities) {
  result <- data.frame(
    id = as.character(source_data$id),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(LABEL_COLS)) {
    label <- LABEL_COLS[i]
    result[[paste0("true_", label)]] <- as.integer(source_data[[label]])
    result[[paste0("prob_", label)]] <- probabilities[, i]
  }

  result
}

val_output <- make_prediction_data(val_data, val_prob)
test_output <- make_prediction_data(test_data, test_prob)

write.csv(
  val_output,
  file.path(OUTPUT_DIR, "vgam_val_predictions.csv"),
  row.names = FALSE
)

write.csv(
  test_output,
  file.path(OUTPUT_DIR, "vgam_test_predictions.csv"),
  row.names = FALSE
)

capture.output(
  summary(vgam_model),
  file = file.path(OUTPUT_DIR, "vgam_summary.txt")
)

cat("\nVGAM selesai.\n")
cat("Model:", file.path(OUTPUT_DIR, "vgam_model.rds"), "\n")
cat("Validasi:", file.path(OUTPUT_DIR, "vgam_val_predictions.csv"), "\n")
cat("Test:", file.path(OUTPUT_DIR, "vgam_test_predictions.csv"), "\n")
