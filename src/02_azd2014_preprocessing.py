from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EXPRESSION_PATH = (DATA_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
DRUG_RESPONSE_PATH = (DATA_DIR / "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")
OUTPUT_PATH = (PROCESSED_DIR / "azd2014_modeling_data.csv")


# ============================================================
# AZD2014 DATA PREPARATION
# ============================================================
#
# Goal:
# Construct a modeling dataset for predicting AZD2014 response
# from baseline RNA-seq expression across cancer cell lines.
#
# Predictors (X):
#   Baseline gene-expression values
#
# Target (y):
#   AZD2014 AUC
#
# One final observation corresponds to one biological cell line
# (DepMap ModelID / depmap_id).
#
# PRISM can contain multiple AZD2014 measurements for the same
# cell line. These measurements are resolved using a predefined
# screen/panel hierarchy before merging with RNA-seq expression.


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOAD DATA")
print("=" * 60)

expression = pd.read_csv(EXPRESSION_PATH)
drug_response = pd.read_csv(DRUG_RESPONSE_PATH)

print("RNA-seq dataset shape:", expression.shape)
print("PRISM dataset shape:", drug_response.shape)


# ============================================================
# 2. EXTRACT AZD2014 MEASUREMENTS
# ============================================================

print("\n" + "=" * 60)
print("AZD2014 PRISM MEASUREMENTS")
print("=" * 60)

azd2014 = drug_response[drug_response["name"] == "AZD2014"].copy()

# Restrict to cell lines for which RNA-seq expression is available.
expression_cell_lines = set(expression["ModelID"].dropna())
azd2014 = azd2014[azd2014["depmap_id"].isin(expression_cell_lines)].copy()
print("AZD2014 measurements:", len(azd2014))
print("Unique AZD2014 cell lines:",azd2014["depmap_id"].nunique())
if azd2014.empty:raise ValueError("No AZD2014 measurements matched the RNA-seq dataset.")


# ============================================================
# 3. INVESTIGATE REPEATED AZD2014 MEASUREMENTS
# ============================================================

# Some cell lines have multiple AZD2014 measurements because they
# were tested in different PRISM screens and/or panels.

azd2014["panel"] = (azd2014["row_name"].astype(str).str.split("_").str[0])
print("\nMeasurements per screen:")
print(azd2014["screen_id"].value_counts())
print("\nMeasurements per panel:")
print(azd2014["panel"].value_counts())

print("\nScreen / panel combinations:")
print(pd.crosstab(azd2014["screen_id"],azd2014["panel"]))
duplicate_cell_lines = (azd2014[azd2014["depmap_id"].duplicated(keep=False)]["depmap_id"].nunique())
print("\nCell lines with repeated PRISM measurements:",duplicate_cell_lines)


# ============================================================
# 4. RESOLVE REPEATED AZD2014 MEASUREMENTS
# ============================================================

# PRISM measurement hierarchy:
#
#   1. MTS010 + PR500
#   2. MTS010 + PR300
#   3. HTS002 + PR500
#
# For cell lines with multiple measurements, retain the
# highest-priority available measurement.

azd2014["priority"] = np.select(
    [
        (
            (azd2014["screen_id"] == "MTS010")
            & (azd2014["panel"] == "PR500")
        ),
        (
            (azd2014["screen_id"] == "MTS010")
            & (azd2014["panel"] == "PR300")
        ),
        (
            (azd2014["screen_id"] == "HTS002")
            & (azd2014["panel"] == "PR500")
        ),
    ],
    [1, 2, 3],
    default=np.nan
)

# Verify that all matched measurements belong to an expected
# screen/panel combination.
unknown_priority = azd2014["priority"].isna().sum()

print("\nMeasurements outside predefined hierarchy:",unknown_priority)

if unknown_priority > 0:
    raise ValueError(
        f"{unknown_priority} AZD2014 measurements could not "
        "be assigned to the predefined PRISM hierarchy."
    )

# Sort from highest to lowest preference and keep one
# measurement per biological cell line.
azd2014_filtered = (azd2014.sort_values("priority").drop_duplicates(subset="depmap_id", keep="first").copy())


# ============================================================
# 5. VERIFY FINAL AZD2014 RESPONSE DATA
# ============================================================

print("\n" + "=" * 60)
print("FILTERED AZD2014 RESPONSE DATA")
print("=" * 60)
print("Rows:", len(azd2014_filtered))
print("Unique cell lines:",azd2014_filtered["depmap_id"].nunique())
print("Duplicated cell lines:", azd2014_filtered["depmap_id"].duplicated().sum())
print("\nSelected measurement priorities:")
print(azd2014_filtered["priority"].value_counts().sort_index())

# Expected result from the original analysis:
#
#   720 cell lines
#   473 MTS010 + PR500
#   237 MTS010 + PR300
#    10 HTS002 + PR500


# ============================================================
# 6. PREPROCESS RNA-SEQ DATA
# ============================================================

print("\n" + "=" * 60)
print("RNA-SEQ PROFILE SELECTION")
print("=" * 60)

azd2014_cell_lines = set(azd2014_filtered["depmap_id"])
expression_azd2014 = expression[expression["ModelID"].isin(azd2014_cell_lines)].copy()

print("RNA-seq profiles before filtering:", len(expression_azd2014))
print("Unique RNA-seq cell lines:", expression_azd2014["ModelID"].nunique())

# Identify cell lines with multiple RNA-seq profiles.
expression_duplicates = expression_azd2014[expression_azd2014["ModelID"].duplicated(keep=False)].copy()
print("Cell lines with multiple RNA-seq profiles:",expression_duplicates["ModelID"].nunique())


# ============================================================
# 7. SELECT ONE RNA-SEQ PROFILE PER CELL LINE
# ============================================================

# DepMap marks a preferred/default molecular profile using
# IsDefaultEntryForMC == "Yes".
#
# Retain the default profile so that each biological cell line
# contributes only one RNA-seq observation.

expression_filtered = expression_azd2014[expression_azd2014["IsDefaultEntryForMC"] == "Yes"].copy()
print("\nRNA-seq profiles after default selection:", len(expression_filtered))
print("Unique cell lines after selection:", expression_filtered["ModelID"].nunique())

print("Duplicated cell lines after selection:",expression_filtered["ModelID"].duplicated().sum())

if expression_filtered["ModelID"].duplicated().any():
    raise ValueError(
        "Duplicate ModelIDs remain after selecting "
        "default RNA-seq profiles."
    )

# Verify that drug-response and RNA-seq datasets contain
# the same biological cell lines.
azd_ids = set(azd2014_filtered["depmap_id"])

expression_ids = set(expression_filtered["ModelID"])
print("\nAZD2014 cell lines without RNA-seq profile:", len(azd_ids - expression_ids))
print("RNA-seq cell lines without AZD2014 response:", len(expression_ids - azd_ids))


# ============================================================
# 8. MERGE RNA-SEQ EXPRESSION WITH AZD2014 RESPONSE
# ============================================================

print("\n" + "=" * 60)
print("MERGE MODELING DATA")
print("=" * 60)

# Remove RNA-seq metadata that should not be used as predictors.
metadata_columns = ["Unnamed: 0", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]

# Drop only columns that actually exist.
metadata_to_drop = [column for column in metadata_columns if column in expression_filtered.columns]

expression_modeling = (expression_filtered .drop(columns=metadata_to_drop) .copy())

# Keep only cell-line ID and AUC target.
azd2014_target = (
    azd2014_filtered[
        ["depmap_id", "auc"]
    ]
    .rename(
        columns={
            "depmap_id": "ModelID"
        }
    )
    .copy()
)

modeling_data = expression_modeling.merge(azd2014_target, on="ModelID", how="inner")

print("Rows:", len(modeling_data))
print("Unique cell lines:", modeling_data["ModelID"].nunique())
print("Missing AUC values:", modeling_data["auc"].isna().sum())
print("Initial modeling-data shape:", modeling_data.shape)


# ============================================================
# 9. FINAL MODELING-DATA QC
# ============================================================

print("\n" + "=" * 60)
print("FINAL QC")
print("=" * 60)

# Lower AUC indicates greater drug sensitivity.
# Higher AUC indicates greater resistance.

print("\nAZD2014 AUC distribution:")
print(modeling_data["auc"].describe())

gene_columns = modeling_data.columns.drop(["ModelID", "auc"])

print("\nNumber of genes before variance filtering:", len(gene_columns))

missing_gene_values = (modeling_data[gene_columns].isna().sum().sum())

print("Missing gene-expression values:", missing_gene_values)

if missing_gene_values > 0:
    raise ValueError(
        f"{missing_gene_values} missing gene-expression "
        "values were found."
    )

# Remove genes with no variation across the selected cell lines.
gene_variance = (modeling_data[gene_columns].var())

zero_variance_genes = (
    gene_variance[
        gene_variance == 0
    ]
    .index)

print("Zero-variance genes:", len(zero_variance_genes))
modeling_data = modeling_data.drop(columns=zero_variance_genes)
final_gene_count = (modeling_data.shape[1] - 2)
print("Final number of gene-expression features:", final_gene_count)
print("Final modeling-data shape:", modeling_data.shape)


# ============================================================
# 10. SAVE FINAL MODELING DATASET
# ============================================================

# Final dataset:
#
#   One row per biological cell line
#   ModelID = cell-line identifier
#   auc     = AZD2014 response target
#   Remaining columns = baseline gene-expression predictors

modeling_data.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print("Saved modeling dataset to:",OUTPUT_PATH)

print(
    "Final dataset:",
    f"{len(modeling_data)} cell lines x "
    f"{final_gene_count} genes"
)