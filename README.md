# GTD Attack Success Prediction: 3CSD Group 8 Implementation

This repository contains the notebook implementation for:

**Evaluating Algorithmic Bias and Feature Thresholds in Predicting Geopolitical Attack Success using Logistic Regression, XGBoost, and a Feedforward Neural Network on the Global Terrorism Database**

## Main Submission File

Submit and run:

```text
notebooks/3CSD_Group 8_Implementation.ipynb
```

Advanced SOTA-oriented alternative:

```text
notebooks/3CSD_Group 8_SOTA_XGBoost_Implementation.ipynb
```

This second notebook uses native-categorical XGBoost, expanded leakage-controlled GTD features, validation hyperparameter search, and validation-optimized thresholds.

The notebook is self-contained. It includes:

- GTD data loading
- feature selection and leakage control
- preprocessing
- temporal train-validation-test splitting
- Logistic Regression baseline
- XGBoost SOTA tabular model
- PyTorch feedforward neural network
- ROC/PR/confusion matrix evaluation
- threshold simulations
- subgroup error disparity analysis
- Logistic Regression coefficient analysis
- XGBoost SHAP or feature-importance analysis

## Dataset

Raw files are expected in:

```text
data-raw/
```

Expected main file:

```text
data-raw/globalterrorismdb_0522dist.xlsx
data-raw/globalterrorismdb_2021Jan-June_1222dist.xlsx
```

Source:

- START Global Terrorism Database: https://www.start.umd.edu/data-tools/GTD
- Kaggle mirror: https://www.kaggle.com/datasets/START-UMD/gtd

The raw data files are ignored by Git because they are large. See `data-raw/README.md` for metadata.

The notebook loads the main 1970-2022 workbook first, then always loads the separate Jan-June 2021 workbook and deduplicates by `eventid`.

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
- `results/tables/subgroup_error_analysis.csv`

## Remaining Helper Code

The `src/` folder now only keeps lightweight preprocessing-related helper files. The required implementation logic is in the notebook.

## LLM Use Disclosure Template

```text
An LLM was used for brainstorming the research framing, organizing the project structure, reviewing possible methodological risks, and improving grammar. All implementation decisions, experimental runs, result interpretation, and final claims were reviewed, verified, and authored by the student researchers.
```
