import pandas as pd
import numpy as np


# ============================================================
# 1. RNA-SEQ EXPRESSION DATA
# ============================================================

# Load DepMap RNA-seq expression data.
# Each row represents an RNA-seq profile and the first six columns
# contain metadata. The remaining columns contain gene-expression
# values reported as log2(TPM + 1).
expression = pd.read_csv( r"data\OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

# --- Basic structure and data quality ---
expression.shape
expression.head()
expression.info()
expression.isna().sum().sort_values(ascending=False).head(20)

# Inspect metadata and gene-expression columns.
expression.columns[:10]
expression.columns[-10:]
expression.iloc[0]

# Gene-expression features begin after the first six metadata columns.
gene_columns = expression.columns[6:]
print("Number of genes:", len(gene_columns))

# Count unique biological cell lines (ModelID).
expression_cell_lines = set(expression["ModelID"].dropna())
print("Unique RNA-seq cell lines:", len(expression_cell_lines))

# Some ModelIDs have more than one RNA-seq profile.
# Inspect these cases and determine which profiles are marked as default.
expression["IsDefaultEntryForMC"].value_counts()
expression_duplicates = expression[expression["ModelID"].duplicated(keep=False)]
expression_duplicates[["ModelID", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC"]].sort_values("ModelID")


# ============================================================
# 2. DEPMAP METADATA
# ============================================================
# Model.csv describes the biological cancer models/cell lines.
# ModelCondition.csv describes culture/experimental conditions.
# OmicsProfiles.csv indexes the available omics experiments.

model = pd.read_csv(r"data\Model.csv")
condition = pd.read_csv(r"data\ModelCondition.csv")
omics = pd.read_csv(r"data\OmicsProfiles.csv")

# --- Model metadata ---
model.shape
model.head()
model.columns

# Explore represented cancer types.
model["OncotreeLineage"].value_counts()
model["OncotreePrimaryDisease"].value_counts().head(20)

# --- Model-condition metadata ---
condition.shape
condition.head()
condition.columns

# --- Omics metadata ---
omics.shape
omics.head()
omics.columns
omics["DataType"].value_counts()


# ============================================================
# 3. PRISM DRUG-RESPONSE DATA
# ============================================================

# Load the PRISM 20Q2 Secondary Repurposing Screen.
# Each row represents a fitted drug-response curve for a particular
# drug/cell-line experimental measurement.
drug_response = pd.read_csv(r"data\prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")

# --- Basic structure ---
drug_response.shape
drug_response.head()
drug_response.columns

# Inspect candidate drug-response variables.
drug_response[["auc", "ec50", "ic50", "r2"]].describe()
drug_response[["auc", "ec50", "ic50", "r2"]].isna().sum()

# Count unique cell lines and drugs.
drug_response["depmap_id"].nunique()
drug_response["broad_id"].nunique()
drug_response["name"].nunique()

# AUC was selected as the regression target because it is available
# for all observations, whereas IC50 has substantial missingness and
# EC50 contains problematic extreme values.

# Determine how many unique cell lines were tested for each drug.
drug_counts = ( drug_response.groupby("name")["depmap_id"].nunique().sort_values(ascending=False))


# ============================================================
# 4. MATCH DRUG-RESPONSE AND RNA-SEQ DATA
# ============================================================

# Identify cell lines represented in each dataset.
drug_cell_lines = set(drug_response["depmap_id"].dropna())

# expression_cell_lines was defined above from expression["ModelID"].

# Find cell lines with both drug-response and RNA-seq data.
overlap_cell_lines = drug_cell_lines.intersection(expression_cell_lines)

print("Drug-response cell lines:", len(drug_cell_lines))       # 737
print("RNA-seq cell lines:", len(expression_cell_lines))       # 1719
print("Overlapping cell lines:", len(overlap_cell_lines))      # 728

# Keep only PRISM measurements from cell lines with RNA-seq data.
drug_response_overlap = drug_response[drug_response["depmap_id"].isin(overlap_cell_lines)].copy()

# For each drug, count the number of unique cell lines that have
# both a drug-response measurement and RNA-seq expression data.
drug_counts_overlap = (drug_response_overlap.groupby("name")["depmap_id"].nunique().sort_values(ascending=False))
drug_counts_overlap.head(20)
drug_counts_overlap.to_csv(r"G:\My Drive\Shimon\stuffs\Data_Science_New_way\Portfolio\Self_projects\Project_3_drug_response_cancer\data\data_I created\drug_counts_overlap.csv")


# ============================================================
# 5. DRUG SELECTION
# ============================================================

# AZD2014 was selected as the first candidate drug because it has
# the largest RNA-seq overlap: 720 unique cell lines.
#
# The first model will use a pan-cancer design:
#   X = baseline RNA-seq gene-expression values
#   y = AZD2014 drug-response AUC
#
# Repeated AZD2014 measurements and the construction of one AUC
# value per cell line will be handled in the preprocessing script.
