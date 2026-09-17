# DepMap Drug Response Prediction

Machine learning prediction of cancer drug response from baseline gene-expression data using **DepMap RNA-seq** and **PRISM drug-response** measurements.

## Project Overview

This project investigates whether baseline gene-expression profiles can predict drug response across cancer cell lines.

The analysis combines:

- DepMap RNA-seq gene-expression data
- PRISM drug-response measurements
- Regularized linear and nonlinear machine-learning models

Drug response is represented by **AUC (area under the dose-response curve)**:

- Lower AUC = greater drug sensitivity
- Higher AUC = greater drug resistance

The project began with **AZD2014 (vistusertib)** as a detailed case study. After finding only weak predictive performance, the workflow was generalized into a reusable **multi-drug screening pipeline** to investigate whether predictability differs across compounds.

---

## Data

The project uses DepMap datasets containing:

- RNA-seq gene-expression profiles
- PRISM 20Q2 secondary drug-response measurements
- Cell-line and molecular-profile metadata

The raw datasets are not included in this repository because of their size.

Derived datasets are generated locally under:

```text
data/processed/
```

### Matching the datasets

RNA-seq and PRISM measurements are matched using DepMap cell-line identifiers (`ModelID` / `depmap_id`).

Initial inspection identified:

- 1,719 unique cell lines in the RNA-seq dataset
- 737 unique cell lines in the PRISM dataset
- 728 cell lines represented in both datasets

The number of matched cell lines was also calculated separately for each drug to identify compounds with sufficient data for modeling.

---

## Data Preprocessing

PRISM can contain multiple measurements for the same drug and biological cell line.

Repeated measurements are resolved using the following predefined hierarchy:

1. `MTS010 + PR500`
2. `MTS010 + PR300`
3. `HTS002 + PR500`

The highest-priority available measurement is retained for each cell line.

Some cell lines also contain multiple RNA-seq profiles. In these cases, the profile designated by DepMap as the default molecular profile:

```text
IsDefaultEntryForMC == "Yes"
```

is retained.

Additional preprocessing includes:

- matching PRISM measurements to available RNA-seq profiles
- retaining one response measurement per cell line
- retaining one RNA-seq profile per cell line
- removing RNA-seq metadata columns
- checking for missing expression values
- removing zero-variance genes

---

## AZD2014 Case Study

**AZD2014 (vistusertib)** was selected as the initial detailed case study because it had high overlap between available PRISM response measurements and RNA-seq profiles.

After preprocessing, the modeling dataset contained:

- **720 cancer cell lines**
- **19,206 gene-expression features**
- **AZD2014 AUC** as the continuous target

This creates a strongly high-dimensional modeling problem in which the number of predictors greatly exceeds the number of observations (`p >> n`).

### Exploratory PCA

Principal component analysis was performed on standardized training-set expression data.

The analysis examined:

- cumulative explained variance
- PC1 versus AZD2014 AUC
- correlations between individual principal components and AUC

No principal component showed a strong relationship with AZD2014 response, suggesting that the major unsupervised axes of global expression variation do not directly capture the response phenotype.

---

## Machine Learning Models

Three regression approaches were evaluated for AZD2014:

### Elastic Net

Elastic Net provides regularization and feature selection, making it suitable for the high-dimensional `p >> n` setting.

Hyperparameters:

- regularization strength (`alpha`)
- L1/L2 mixing parameter (`l1_ratio`)

were optimized using 5-fold cross-validation.

### Support Vector Regression

RBF-SVR was used to test whether nonlinear relationships between gene expression and drug response could improve prediction over the linear Elastic Net model.

### Random Forest

Random Forest regression was used as a nonlinear tree-based alternative.

Hyperparameters including the number of trees, number of features considered at each split, tree depth, and minimum leaf size were tuned using cross-validation.

---

## AZD2014 Results

An 85/15 train/test split was used with:

```text
random_state = 101
```

Hyperparameters were selected using **5-fold cross-validation on the training data**, optimizing RMSE.

A mean-only prediction was also used as a reference baseline.

| Model | Test RMSE | Test R² |
|---|---:|---:|
| Elastic Net | ~0.093 | ~0.029 |
| RBF-SVR | 0.0939 | 0.0169 |
| Random Forest | 0.0911 | 0.0747 |

### Elastic Net

Cross-validation selected approximately:

```text
alpha = 0.1
l1_ratio = 0.1
```

Only **38 of 19,206 gene-expression coefficients** remained non-zero.

The resulting model was sparse but explained little of the variation in AZD2014 response.

### SVR

RBF-SVR did not materially improve predictive performance over Elastic Net.

### Random Forest

Random Forest produced the highest held-out R² among the three models:

```text
Test R²  ≈ 0.075
Train R² ≈ 0.871
```

The large difference between training and test performance indicates substantial overfitting.

### AZD2014 Conclusion

Overall, baseline gene expression alone showed **limited predictive ability for AZD2014 response** in this pan-cancer dataset.

Rather than treating this as an endpoint, the weak result motivated the next stage of the project:

> Does predictability from baseline gene expression differ between drugs?

---

## Generalized Drug-Screening Pipeline

The AZD2014 workflow was generalized into the reusable:

```python
screen_drug()
```

function.

For any selected PRISM compound, the function automatically performs:

1. drug-response extraction
2. RNA-seq / PRISM cell-line matching
3. repeated PRISM measurement resolution
4. default RNA-seq profile selection
5. metadata removal
6. zero-variance gene filtering
7. AUC distribution analysis
8. train/test splitting
9. mean-only baseline fitting
10. hyperparameter tuning
11. model evaluation
12. diagnostic plotting
13. automatic result saving

Models can be enabled independently:

```python
screen_drug(
    expression,
    drug_response,
    drug_name,
    enet=True,
    svr=True,
    rf=False
)
```

This allows relatively fast initial screening before applying more computationally expensive models to selected drugs.

---

## Evaluation Metrics

Each model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Test R²
- Cross-validation RMSE
- Training R²
- RMSE relative to test-set AUC standard deviation
- RMSE improvement relative to the mean-only baseline

Training and test R² are reported together to help identify overfitting.

---

## Multi-Drug Screening

The standardized pipeline is applied to multiple drugs with relatively high overlap between PRISM measurements and available RNA-seq profiles.

The current screening set includes:

- AZD2014
- cinacalcet
- osimertinib
- teniposide
- floxuridine
- tanespimycin
- ponatinib
- BMS-754807
- pevonedistat
- talazoparib
- trametinib
- alvespimycin
- everolimus

The same preprocessing, train/test strategy, cross-validation procedure, hyperparameter grids, and evaluation metrics are used across drugs.

The purpose of this stage is **exploratory screening**, rather than extensive optimization of every compound.

Drugs showing stronger and more consistent predictive performance can subsequently be selected for deeper analysis.

---

## Project Structure

```text
Project_3_drug_response_cancer/
│
├── data/
│   └── processed/
│
├── docs/
│   └── Drug_Response_Prediction_Project_Summary.docx
│
├── figures/
│
├── results/
│   └── drug_screening/
│
├── src/
│   ├── 01_data_inspection.py
│   ├── 02_azd2014_preprocessing.py
│   ├── 03_azd2014_modeling.py
│   ├── 04_multi_drug_screening.py
│   └── drug_screening.py
│
├── .gitignore
├── environment.yml
└── README.md
```

### Source Files

**`01_data_inspection.py`**

Initial inspection of the DepMap and PRISM datasets, evaluation of RNA-seq/PRISM overlap, and identification of drugs with sufficient matched cell lines.

**`02_azd2014_preprocessing.py`**

Detailed preprocessing of AZD2014 measurements and construction of the final AZD2014 modeling dataset.

**`03_azd2014_modeling.py`**

Detailed AZD2014 analysis including PCA, Elastic Net, RBF-SVR, Random Forest, model evaluation, and diagnostic plots.

**`drug_screening.py`**

Reusable preprocessing and machine-learning pipeline for standardized drug screening.

**`04_multi_drug_screening.py`**

Applies the standardized screening pipeline across multiple selected drugs and aggregates the results.

---

## Running the Project

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate drug_response
```

Run the analysis from the project root:

```bash
python src/01_data_inspection.py
python src/02_azd2014_preprocessing.py
python src/03_azd2014_modeling.py
python src/04_multi_drug_screening.py
```

---

## Outputs

Detailed AZD2014 results are stored under:

```text
results/
figures/
```

The generalized screening pipeline creates a separate directory for each drug:

```text
results/drug_screening/<drug_name>/
```

Each drug analysis can contain:

```text
data_summary.csv
model_results.csv
test_set_summary.csv
auc_distribution.png

Elastic_Net/
    results.txt
    predictions.csv
    observed_vs_predicted.png
    residuals_vs_predicted.png

SVR/
    results.txt
    predictions.csv
    observed_vs_predicted.png
    residuals_vs_predicted.png
```

Random Forest outputs are also generated when that model is enabled.

---

## Methodological Considerations

Several aspects of the analysis were designed to reduce common modeling problems:

- Scaling is performed inside scikit-learn pipelines.
- Hyperparameter selection is performed using training-set cross-validation.
- The held-out test set is not used for hyperparameter optimization.
- A mean-only baseline provides a reference for evaluating whether expression features add predictive information.
- Training and test performance are both reported to expose overfitting.
- The same workflow is applied across drugs during screening.

---

## Limitations

This analysis has several important limitations.

### High dimensionality

The number of expression features (~19,000 genes) greatly exceeds the number of available cell lines.

### Pan-cancer analysis

Cancer lineage may contribute to both gene-expression variation and drug response. The current analysis does not explicitly model tissue or lineage effects.

### Single molecular modality

Baseline RNA-seq represents only one component of cancer-cell biology.

Drug response may also depend on:

- mutations
- copy-number alterations
- proteomics
- pathway activity
- cancer lineage
- other molecular features

### Exploratory multi-drug screening

The multi-drug stage compares many compounds using the same modeling pipeline.

Therefore, drugs selected because they perform well during this screening stage should ideally undergo a new validation procedure before their performance is interpreted as confirmatory.

### Biological interpretation

Genes selected by Elastic Net or other predictive models represent statistical associations and should not automatically be interpreted as causal determinants of drug response.

---

## Potential Extensions

Future development could include:

- pathway-level expression features
- supervised dimensionality reduction
- lineage-aware modeling
- mutation data
- copy-number alterations
- proteomic features
- multimodal molecular models
- repeated or nested cross-validation
- deeper analysis of drugs showing stronger predictive signal

---

## Summary

This project demonstrates an end-to-end computational biology and machine-learning workflow for predicting cancer drug response from high-dimensional molecular data.

The workflow includes:

- integration of large public biological datasets
- biological sample matching
- resolution of repeated experimental measurements
- high-dimensional RNA-seq preprocessing
- regularized linear regression
- nonlinear kernel methods
- tree-based modeling
- cross-validation and hyperparameter tuning
- baseline comparison
- overfitting assessment
- model diagnostics
- reusable multi-drug screening

The weak predictive performance observed for AZD2014 is retained as part of the scientific result rather than optimized away. It motivated the development of the broader multi-drug screening framework and highlights the limitations of predicting complex drug-response phenotypes from baseline gene expression alone.