# GTD Attack Success Prediction: 3CSD Group 8

This repository contains the final cleaned project files for predicting the GTD-coded `success` outcome of recorded terrorism incidents.

## Main Notebooks

```text
notebooks/3CSD_Group 8_Implementation.ipynb
notebooks/3CSD_Group 8_Novel_DL_Implementation.ipynb
```

The main implementation notebook is self-contained and includes data loading, cleaning, temporal/random split triggers, model training, scoring, confusion matrices, learning curves, and permutation feature importance.

The novel deep-learning notebook contains the proposed **GTD-FocalResNet** experiment.

## Main Documents

```text
paper/ML_Project_Working_Document.docx
paper/GTD_Attack_Success_Manuscript.docx
paper/GTD_Model_Notebook_Summary_for_Groupmates.docx
paper/Template_ML_Project.docx
```

The template file is retained for formatting reference.

## Dataset

Raw GTD files are expected in:

```text
data/raw/
```

Expected files:

```text
data/raw/globalterrorismdb_0522dist.xlsx
data/raw/globalterrorismdb_2021Jan-June_1222dist.xlsx
```

Source:

- START Global Terrorism Database: https://www.start.umd.edu/data-tools/GTD

## Processed Data

The notebook uses two kinds of processed data:

```text
data/processed/gtd_light_full.pkl
```

This is the cleaned selected-column cache used to avoid reparsing the raw Excel files on reruns.

The actual train/validation/test split files are saved under:

```text
data/processed/temporal/
data/processed/random/
```

Each split folder contains:

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

The temporal split is the main paper setting. The random split is only for comparison.

## Main Notebook Settings

Important switches inside the notebook:

```python
SPLIT_MODE = "temporal"          # "temporal" or "random"
TRAINING_PROFILE = "submission"  # "submission" or "max"
CACHE_DATA = True
SAVE_PROCESSED_SPLITS = True
REBUILD_DATA_CACHE = False
DOWNLOAD_DATA_IF_MISSING = False
```

## Models

The final implementation compares:

- Logistic Regression
- CART
- XGBoost
- Feedforward Neural Network
- GTD-FocalResNet

## Current Outputs

Current outputs are saved under:

```text
results/light_reproducible/
models/light_reproducible/
```

The paper mainly uses:

```text
results/light_reproducible/temporal/tables/scores.csv
results/light_reproducible/temporal/figures/confusion_matrices.png
results/light_reproducible/temporal/figures/learning_curve_neural_models_f1_roc_auc.png
results/light_reproducible/temporal/figures/feature_importance_all_models.png
results/tables/temporal_vs_random_split_comparison.csv
results/figures/temporal_vs_random_split_comparison.png
```

## Environment

Using the existing `ml-course` environment:

```powershell
conda activate ml-course
python -m pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
notebooks/3CSD_Group 8_Implementation.ipynb
```

## LLM Use Disclosure

```text
An LLM was used for brainstorming the research framing, organizing the project structure, reviewing possible methodological risks, improving grammar, and assisting with document/notebook cleanup. All implementation decisions, experimental runs, result interpretation, and final claims were reviewed, verified, and authored by the student researchers.
```
