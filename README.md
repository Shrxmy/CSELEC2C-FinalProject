# GTD Attack Success Prediction: 3CSD Group 8 Implementation

This repository contains the notebook implementation for:

**Evaluating Algorithmic Bias and Feature Thresholds in Predicting Geopolitical Attack Success using Logistic Regression, CART, XGBoost, and a Feedforward Neural Network on the Global Terrorism Database**

## Main Submission File

Submit and run:

```text
notebooks/3CSD_Group 8_Implementation.ipynb
```

This second notebook uses native-categorical XGBoost, expanded leakage-controlled GTD features, validation hyperparameter search, and validation-optimized thresholds.

The previous three-model notebook is retained as:

```text
notebooks/3CSD_Group 8_Implementation_Original.ipynb
```

A self-contained submission copy is available if only the notebook and PDF are required:

```text
notebooks/3CSD_Group 8_Implementation_Self-Contained.ipynb
```

This self-contained version includes the helper pipeline code directly inside the notebook and does not require the `src/` folder. It still requires the GTD raw files and Python packages from `requirements.txt`.

A random-split comparison notebook is also available for checking temporal-vs-random split performance:

```text
notebooks/3CSD_Group 8_Implementation_Random-Split.ipynb
```

The random-split notebook writes outputs to `models/random_split/`, `results/figures/random_split/`, and `results/tables/random_split/` so it does not overwrite the manuscript-facing temporal-split results.

Current random-vs-temporal comparison outputs are also saved as:

```text
results/tables/temporal_vs_random_split_comparison.csv
results/figures/temporal_vs_random_split_comparison.png
```

Summary: the random split produces higher test scores for all models because it mixes years across train/validation/test. The temporal split remains the main manuscript setting because it is stricter and better approximates forward-looking generalization.

The notebook is self-contained. It includes:

- GTD data loading
- feature selection and leakage control
- preprocessing
- temporal train-validation-test splitting in the main notebook
- optional stratified random split in the comparison notebook
- Logistic Regression linear baseline
- CART single-tree baseline
- XGBoost SOTA tabular model
- PyTorch feedforward neural network
- ROC/PR/confusion matrix evaluation
- threshold simulations
- subgroup error disparity analysis
- Logistic Regression coefficient analysis
- XGBoost SHAP or feature-importance analysis

## Dataset

Raw files are expected in the canonical raw-data directory:

```text
data/raw/
```

Expected files:

```text
data/raw/globalterrorismdb_0522dist.xlsx
data/raw/globalterrorismdb_2021Jan-June_1222dist.xlsx
```

The main notebook expects both GTD files in `data/raw/`. The 2021 supplement is mandatory because the provider distributes that portion separately. Accepted formats are `.xlsx`, `.csv`, or `.parquet` using the expected GTD filenames.

Source:

- START Global Terrorism Database: https://www.start.umd.edu/data-tools/GTD
- Kaggle mirror: https://www.kaggle.com/datasets/START-UMD/gtd

The raw data files are ignored by Git because they are large. See `data/raw/README.md` for metadata.

The notebook loads the main GTD file and the mandatory separate Jan-June 2021 supplement, then deduplicates by `eventid`.

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

## Notebook Settings

Inside the notebook:

```python
USE_SAMPLE = False
RUN_SHAP = True
USE_GPU = True
SAVE_PROCESSED_SPLITS = False
```

Use `USE_SAMPLE = True` only for a quick test. Final reported results should use `USE_SAMPLE = False`.

`SAVE_PROCESSED_SPLITS = False` avoids writing extra Parquet/CSV split files. Set it to `True` only if the instructor wants intermediate processed data saved for reproducibility.

Set `USE_GPU = True` when the machine has a CUDA-capable GPU. GPU acceleration applies to XGBoost and the PyTorch feedforward neural network. Logistic Regression remains CPU-based.

On Windows or WSL2, PyTorch GPU support depends on the installed PyTorch build, NVIDIA driver, and CUDA availability. The notebook prints `torch.cuda.is_available()` and the selected neural-network device in the setup cell.

XGBoost is also requested on CUDA when available. During evaluation, it may show a warning because the sklearn preprocessing output is a CPU matrix; training still uses the requested XGBoost device.

## Outputs

The notebook saves artifacts to:

```text
models/
results/figures/
results/tables/
data/processed/
```

Important output tables:

- `results/tables/model_metrics.csv`
- `results/tables/threshold_simulation_results.csv`
- `results/tables/proposal_validation_selected_thresholds.csv`
- `results/tables/subgroup_error_analysis.csv`
- `results/tables/proposal_subgroup_disparity_summary.csv`
- `results/tables/proposal_logistic_coefficients.csv`
- `results/tables/proposal_xgboost_importance.csv`
- `results/tables/proposal_xgboost_learning_curves.csv`
- `results/tables/proposal_strict_feature_audit.csv`
- `results/tables/proposal_feature_group_ablation.csv`
- `results/tables/proposal_bootstrap_confidence_intervals.csv`
- `results/tables/proposal_calibration_curves.csv`

## Reusable Helper Code

The `src/` folder contains reusable project helpers, especially `src/gtd_pipeline.py`, so the main notebook stays concise and reproducible.

## LLM Use Disclosure

```text
An LLM was used for brainstorming the research framing, organizing the project structure, reviewing possible methodological risks, and improving grammar. All implementation decisions, experimental runs, result interpretation, and final claims were reviewed, verified, and authored by the student researchers.
```
