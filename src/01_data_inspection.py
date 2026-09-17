from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXPRESSION_PATH = (DATA_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
MODEL_PATH = DATA_DIR / "Model.csv"
CONDITION_PATH = DATA_DIR / "ModelCondition.csv"
OMICS_PATH = DATA_DIR / "OmicsProfiles.csv"
DRUG_RESPONSE_PATH = (DATA_DIR / "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")
DRUG_COUNTS_OUTPUT = (PROCESSED_DIR / "drug_counts_overlap.csv")


# ============================================================
# 1. RNA-SEQ EXPRESSION DATA
# ============================================================

print("\n" + "=" * 60)
print("RNA-SEQ EXPRESSION DATA")
print("=" * 60)

# Load DepMap RNA-seq expression data.
#
# Each row represents an RNA-seq profile.
# The first six columns contain metadata and the remaining columns
# contain gene-expression values reported as log2(TPM + 1).

expression = pd.read_csv(EXPRESSION_PATH)

print("Expression shape:", expression.shape)
print("\nFirst five rows:")
print(expression.head())
print("\nFirst 10 columns:")
print(expression.columns[:10].tolist())
print("\nLast 10 columns:")
print(expression.columns[-10:].tolist())

# Gene-expression features begin after the first six metadata columns.
gene_columns = expression.columns[6:]
print("\nNumber of gene-expression features:", len(gene_columns))

# Inspect missing values.
missing_expression = (expression.isna().sum().sort_values(ascending=False))
print("\nColumns with the most missing values:")
print(missing_expression.head(20))

# Count unique biological cell lines.
expression_cell_lines = set(expression["ModelID"].dropna())
print("\nUnique RNA-seq cell lines:",len(expression_cell_lines))

# Some ModelIDs have more than one RNA-seq profile.
expression_duplicates = expression[expression["ModelID"].duplicated(keep=False)].copy()
print("Cell lines with multiple RNA-seq profiles:",expression_duplicates["ModelID"].nunique())

print("\nDefault-entry status:")
print(expression["IsDefaultEntryForMC"].value_counts(dropna=False))


# ============================================================
# 2. DEPMAP METADATA
# ============================================================

print("\n" + "=" * 60)
print("DEPMAP METADATA")
print("=" * 60)

# Model.csv describes the biological cancer models/cell lines.
# ModelCondition.csv describes culture/experimental conditions.
# OmicsProfiles.csv indexes the available omics experiments.

model = pd.read_csv(MODEL_PATH)
condition = pd.read_csv(CONDITION_PATH)
omics = pd.read_csv(OMICS_PATH)
print("Model metadata shape:", model.shape)
print("Model-condition metadata shape:", condition.shape)
print("Omics metadata shape:", omics.shape)

# Explore represented cancer types.
print("\nMost common OncoTree lineages:")
print(model["OncotreeLineage"].value_counts().head(20))
print("\nMost common primary diseases:")
print(model["OncotreePrimaryDisease"].value_counts().head(20))
print("\nOmics data types:")
print(omics["DataType"].value_counts())


# ============================================================
# 3. PRISM DRUG-RESPONSE DATA
# ============================================================

print("\n" + "=" * 60)
print("PRISM DRUG-RESPONSE DATA")
print("=" * 60)

# Load the PRISM 20Q2 Secondary Repurposing Screen.
#
# Each row represents a fitted drug-response curve for a particular
# drug/cell-line experimental measurement.

drug_response = pd.read_csv(DRUG_RESPONSE_PATH)

print("Drug-response shape:", drug_response.shape)
print("\nFirst five rows:")
print(drug_response.head())

# Inspect candidate drug-response variables.
response_variables = ["auc", "ec50", "ic50", "r2"]

print("\nDrug-response variable summary:")
print(drug_response[response_variables].describe())

print("\nMissing drug-response values:")
print(drug_response[response_variables].isna().sum())

# Count unique cell lines and compounds.
print("\nUnique drug-response cell lines:", drug_response["depmap_id"].nunique())

print("Unique Broad compound IDs:", drug_response["broad_id"].nunique())

print("Unique drug names:", drug_response["name"].nunique())

# AUC is selected as the regression target because it is available
# across the PRISM measurements used here, whereas IC50 has substantial
# missingness and EC50 contains problematic extreme values.

# Determine how many unique cell lines were tested for each drug.
drug_counts = (drug_response.groupby("name")["depmap_id"].nunique().sort_values(ascending=False))
print("\nDrugs with the largest number of tested cell lines:")
print(drug_counts.head(20))


# ============================================================
# 4. MATCH DRUG-RESPONSE AND RNA-SEQ DATA
# ============================================================

print("\n" + "=" * 60)
print("RNA-SEQ / PRISM OVERLAP")
print("=" * 60)

# Identify cell lines represented in the PRISM dataset.
drug_cell_lines = set(drug_response["depmap_id"].dropna())

# Find cell lines represented in both datasets.
overlap_cell_lines = (drug_cell_lines.intersection(expression_cell_lines))

print("Drug-response cell lines:",len(drug_cell_lines))
print("RNA-seq cell lines:",len(expression_cell_lines))
print("Overlapping cell lines:",len(overlap_cell_lines))

# Keep PRISM measurements only for cell lines with RNA-seq data.
drug_response_overlap = drug_response[drug_response["depmap_id"].isin(overlap_cell_lines)].copy()

# For each drug, count the number of unique cell lines with both:
#
#   1. PRISM drug-response measurements
#   2. RNA-seq expression data

drug_counts_overlap = (drug_response_overlap.groupby("name")["depmap_id"].nunique().sort_values(ascending=False))
print("\nDrugs with the largest RNA-seq / PRISM overlap:")
print(drug_counts_overlap.head(20))

# Save for use during drug selection and multi-drug screening.
drug_counts_overlap.to_csv(DRUG_COUNTS_OUTPUT,header=["matched_cell_lines"])
print("\nSaved overlap counts to:",DRUG_COUNTS_OUTPUT)


# ============================================================
# 5. DRUG SELECTION
# ============================================================

print("\n" + "=" * 60)
print("INITIAL DRUG SELECTION")
print("=" * 60)

# AZD2014 was selected as the initial case study because it has
# one of the largest RNA-seq / PRISM overlaps.
#
# The initial modeling design is pan-cancer:
#
#   X = baseline RNA-seq gene-expression values
#   y = AZD2014 drug-response AUC
#
# Repeated PRISM measurements and duplicate RNA-seq profiles are
# resolved in the AZD2014 preprocessing script.

azd_overlap = drug_counts_overlap.get("AZD2014", 0)

print("AZD2014 matched cell lines:",azd_overlap)

print(
    "\nNext step:"
    "\nRun 02_azd2014_preprocessing.py to construct "
    "the AZD2014 modeling dataset.")