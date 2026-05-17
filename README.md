# GTD-FocalResNet: A Residual Focal-Loss Neural Model for Attack Success Prediction

This repository contains the final machine-learning project for predicting the Global Terrorism Database (GTD) `success` outcome of recorded terrorist incidents. The study compares classical, tree-based, boosted, and neural tabular models under a forward-looking temporal validation design. It also proposes **GTD-FocalResNet**, a residual focal-loss neural architecture adapted for imbalanced GTD attack-success prediction.

## Project Information

**Course and Section:** 3CSD – Machine Learning  
**Group:** Group 8  
**Members:** Abelardo, Aquino, Balingit, and Gumban

## Final Submission Files

```text
paper/3CSD_Group 8_Manuscript.docx
paper/3CSD_Group 8_Manuscript.pdf
paper/3CSD_Group 8_Implementation.pdf
notebooks/3CSD_Group 8_Implementation.ipynb
notebooks/3CSD_Group 8_Novel_DL_Implementation.ipynb
```

The main notebook is self-contained and includes data loading, cleaning, split processing, model training, evaluation, visualizations, and model saving. The separate novel deep-learning notebook documents the GTD-FocalResNet development and tuning process.

## Study Design

The primary evaluation uses a **temporal split** to approximate forward-looking prediction. A **random split** is included only as a comparison because random splitting mixes years across train, validation, and test sets and usually produces easier results.

Final model set:

1. Logistic Regression
2. CART
3. XGBoost
4. Feedforward Neural Network
5. GTD-FocalResNet

Main outputs include score tables, confusion matrices, neural learning curves, permutation feature importance, saved model artifacts, and temporal-versus-random split comparison results.

## Dataset

The project uses the Global Terrorism Database from START:

```text
https://www.start.umd.edu/data-tools/GTD
```

Raw GTD files should be placed in:

```text
data/raw/
```

Expected raw files:

```text
data/raw/globalterrorismdb_0522dist.xlsx
data/raw/globalterrorismdb_2021Jan-June_1222dist.xlsx
```

If raw files are missing, the notebook includes optional direct-link download logic. By default, downloading is disabled for faster and safer reruns.

## Processed Data

The notebook uses a cleaned cache to avoid repeatedly parsing large Excel files:

```text
data/processed/gtd_light_full.pkl
```

The actual train/validation/test split files are saved separately:

```text
data/processed/temporal/
data/processed/random/
```

Each split directory contains:

```text
X_train.parquet
X_valid.parquet
X_test.parquet
y_train.csv
y_valid.csv
y_test.csv
split_summary.csv
numeric_features.csv
categorical_features.csv
```

## Main Results and Artifacts

Important result files:

```text
results/light_reproducible/temporal/tables/scores.csv
results/light_reproducible/random/tables/scores.csv
results/tables/temporal_vs_random_split_comparison.csv
```

Important figures:

```text
results/light_reproducible/temporal/figures/confusion_matrices.png
results/light_reproducible/temporal/figures/learning_curve_neural_models_f1_roc_auc.png
results/light_reproducible/temporal/figures/feature_importance_all_models.png
results/figures/temporal_vs_random_split_comparison.png
```

Saved model artifacts are stored in:

```text
models/light_reproducible/
```

PyTorch models are saved as `.pt` checkpoints with matching preprocessing files, while scikit-learn and XGBoost models are saved as `.joblib` files.

## Reproducing the Main Notebook

Use the `ml-course` environment or install the required dependencies:

```powershell
conda activate ml-course
python -m pip install -r requirements.txt
jupyter notebook
```

Open:

```text
notebooks/3CSD_Group 8_Implementation.ipynb
```

Recommended settings for reproducing the main temporal run:

```python
SPLIT_MODE = "temporal"
TRAINING_PROFILE = "submission"  # use "max" for the longer final run
CACHE_DATA = True
SAVE_PROCESSED_SPLITS = True
REBUILD_DATA_CACHE = False
DOWNLOAD_DATA_IF_MISSING = False
```

To reproduce the random-split comparison, change:

```python
SPLIT_MODE = "random"
```

## Notebook PDF Export

The implementation notebook has already been exported as:

```text
paper/3CSD_Group 8_Implementation.pdf
```

If re-exporting with nbconvert, ensure that `pandoc` and a LaTeX engine such as `tectonic` are available from the active environment.

## Notes on Interpretation

The paper treats GTD attack success as a coded database outcome, not as a complete measure of political, social, or strategic impact. Results should therefore be interpreted as predictive patterns in recorded GTD data rather than causal explanations of terrorism outcomes.

## LLM Use Disclosure

An LLM was used to assist with research framing, project organization, grammar review, document formatting, and notebook cleanup. All implementation decisions, experimental runs, result interpretation, and final claims were reviewed and verified by the student researchers.
