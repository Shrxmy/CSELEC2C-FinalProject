# Evaluating Algorithmic Bias and Feature Thresholds in Predicting Geopolitical Attack Success Using Logistic Regression, XGBoost, and a Feedforward Neural Network on the Global Terrorism Database

## Abstract

This study investigates how machine learning models predict attack success in the Global Terrorism Database (GTD), with emphasis on model assumptions, internal feature weighting, imbalanced classification, and decision-threshold sensitivity. We compare a class-weighted logistic regression baseline with two nonlinear alternatives: Extreme Gradient Boosting (XGBoost) and a feedforward neural network. Rather than treating predictive performance as the sole objective, the paper evaluates how each model distributes errors across geopolitical and tactical subgroups such as region, country, attack type, target type, and weapon type. We use a temporal train-validation-test design to approximate forward-looking deployment and avoid synthetic test data. Interpretability is assessed through coefficient analysis and SHAP-style feature attribution, while threshold simulations examine how operational decisions change when the classification boundary is shifted.

## 1. Introduction

Machine learning systems are increasingly applied to conflict, security, and geopolitical risk analysis. These settings are technically difficult and ethically sensitive because data are historically uneven, reporting practices vary by country and period, and predictive errors may be distributed unequally across regions or tactical contexts. The GTD provides a large empirical record of terrorist incidents, making it useful for studying both predictive modeling and the limitations of algorithmic reasoning under social and geopolitical uncertainty.

This project asks how classical and nonlinear supervised learning models differ in predicting whether a recorded attack is successful, and how their decisions change under different probability thresholds. The central interest is not only which model obtains the highest score. The analysis studies what the models learn, which features they emphasize, how class imbalance affects predictions, and whether false-positive or false-negative rates vary across geopolitical and tactical groups.

## 2. Data

The project uses the Global Terrorism Database from START, placed locally under `data/raw/`. Both the main GTD workbook and the separate Jan-June 2021 supplement are required because the dataset provider distributes them as separate files. The target variable is `success`, coded as a binary indicator. The modeling frame uses only variables plausibly available at or near attack design or event recording time, including year, month, region, country, attack type, target type, weapon type, suicide indicator, multiple-incident indicator, claim indicator, and related categorical descriptors. Casualty, property-damage, ransom, and hostage-outcome fields are intentionally excluded because they are post-event consequences and would create target leakage.

The split is temporal:

- Training: events through 2014
- Validation: 2015 to 2017
- Testing: 2018 onward

This design is stricter than a random split because the model is evaluated on later historical periods rather than interpolating across the same period.

## 3. Methods

### 3.1 Logistic Regression

Logistic regression is used as the interpretable baseline. It models the log-odds of success as a linear function of standardized numeric variables and one-hot encoded categorical variables. Class weighting is used to reduce the effect of class imbalance. Its main advantage is interpretability: coefficients indicate the direction and relative strength of feature associations under the linear-additive assumption.

### 3.2 XGBoost

XGBoost is used as the tree-based nonlinear model. It can model interactions between variables such as weapon type, region, and target type without manually specifying those interactions. The implementation uses gradient-boosted decision trees with a logistic objective and class-imbalance weighting. Its assumptions differ from logistic regression because it partitions the feature space and aggregates many weak learners rather than fitting a single global linear boundary.

### 3.3 Feedforward Neural Network

The neural network is implemented as a multilayer perceptron with two hidden layers. It provides a nonlinear function approximator that can model complex interactions after preprocessing. It is included to compare whether additional representational flexibility produces meaningful improvements or instead creates less interpretable behavior without clear performance gains.

## 4. Interpretability and Bias Analysis

The project operationalizes algorithmic bias as unequal classification error behavior across subgroups. For each model, false-positive and false-negative rates are computed by region, country, attack type, target type, and weapon type, subject to a minimum group size. This does not claim that the dataset contains ground-truth social fairness labels. Instead, it measures whether the predictive system behaves unevenly across geopolitical and tactical contexts.

Feature influence is studied with logistic regression coefficient magnitudes and SHAP or feature-importance plots for nonlinear models. These outputs help answer whether models rely mainly on broad geopolitical fields, tactical variables, or specific weapon and target categories.

## 5. Threshold Simulation

The default classification threshold of 0.50 is not treated as neutral. The pipeline evaluates thresholds from 0.30 to 0.70 and records precision, recall, F1 score, false-positive rate, and false-negative rate. This matters because a risk model deployed with a lower threshold may increase recall while also increasing false alarms, and a higher threshold may reduce false positives while missing more successful attacks.

## 6. Reproducibility

The implementation is organized around the main notebook `notebooks/3CSD_Group 8_Implementation.ipynb`. To reproduce the analysis:

```bash
pip install -r requirements.txt
jupyter notebook
```

Then place the GTD files in `data/raw/`, open the main notebook, set `USE_SAMPLE = False`, and run all cells. For a quick smoke test, set `USE_SAMPLE = True` inside the notebook before running. The notebook writes trained models to `models/`, tables to `results/tables/`, and figures to `results/figures/`.

## 7. Expected Results Tables and Figures

The manuscript should report:

- `model_metrics.csv`: ROC-AUC, average precision, F1, recall, precision, balanced accuracy
- `proposal_robust_model_metrics.csv`: imbalance-aware metrics including failure-class AP, MCC, and Brier score
- `proposal_bootstrap_confidence_intervals.csv`: bootstrap confidence intervals for key test metrics
- `proposal_validation_selected_thresholds.csv`: validation-selected threshold evaluation
- `threshold_simulation_results.csv`: model behavior under thresholds 0.30 to 0.70
- `subgroup_error_analysis.csv`: false-positive and false-negative rates by subgroup
- `proposal_subgroup_disparity_summary.csv`: compact subgroup disparity gaps
- `proposal_feature_group_ablation.csv`: feature-group ablation results
- `proposal_strict_feature_audit.csv`: leakage-sensitivity audit
- ROC, precision-recall, confusion matrix, threshold-sensitivity, subgroup-disparity, and interpretability figures

## 8. Limitations

The GTD reflects reported and coded incidents, so the model learns from historical data collection patterns as well as from the underlying geopolitical phenomenon. Success is also a simplified binary outcome that may not capture strategic, symbolic, or long-term consequences. Subgroup disparity analysis should therefore be interpreted as model-behavior analysis, not as a direct causal claim about regions, countries, or groups. Finally, strong predictive performance does not imply policy suitability; the model is designed for methodological study and interpretability, not operational deployment.

## 9. LLM Use Disclosure

An LLM was used for brainstorming the research framing, organizing the project structure, reviewing possible methodological risks, and improving grammar. All implementation decisions, experimental runs, result interpretation, and final claims should be reviewed, verified, and authored by the student researchers.

