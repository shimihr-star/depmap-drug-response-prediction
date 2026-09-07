import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# AZD2014 (vistusertib) is an experimental anti-cancer drug that inhibits mTOR,
# specifically both the mTORC1 and mTORC2 signaling complexes. The mTOR pathway
# regulates important cellular processes including growth, proliferation, metabolism,
# and survival, and is frequently dysregulated in cancer.
# AZD2014 has been investigated in clinical trials for several cancer types.
# Therefore, variation in gene expression across cell lines may contain molecular
# features associated with sensitivity or resistance to AZD2014.


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
# One final observation should correspond to one biological
# cell line (DepMap ModelID / depmap_id).
#
# PRISM may contain multiple AZD2014 measurements for the same
# cell line. These measurements must therefore be resolved before
# merging drug response with RNA-seq expression.


# ============================================================
# 1. LOAD DATA
# ============================================================
expression = pd.read_csv( r"data\OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
drug_response = pd.read_csv( r"data\prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")

# ============================================================
# 2. EXTRACT AZD2014 MEASUREMENTS
# ============================================================
azd2014 = drug_response[drug_response["name"] == "AZD2014"].copy()

# Restrict to cell lines for which RNA-seq expression is available.
expression_cell_lines = set(expression["ModelID"].dropna())
azd2014 = azd2014[azd2014["depmap_id"].isin(expression_cell_lines)].copy()
print("AZD2014 measurements:", len(azd2014))
print("Unique cell lines:", azd2014["depmap_id"].nunique())

# ============================================================
# 3. INVESTIGATE REPEATED AZD2014 MEASUREMENTS
# ============================================================
# There are 1,191 AZD2014 measurements but only 720 unique
# cell lines, meaning that some cell lines have multiple
# drug-response measurements.

azd2014["depmap_id"].value_counts().head(20)

# Repeated measurements arise from different PRISM screens and/or
# PR300/PR500 panels.
#
# Selection hierarchy based on the PRISM documentation:
#
#   1. MTS010 + PR500  -> preferred
#   2. MTS010 + PR300  -> use if PR500 is unavailable
#   3. HTS002          -> use if MTS010 is unavailable
#
# Before applying this hierarchy, verify that the AZD2014 data
# contain the expected screen/panel combinations.
azd2014["screen_id"].value_counts()
azd2014["row_name"].str.split("_").str[0].value_counts()
pd.crosstab(azd2014["screen_id"],azd2014["row_name"].str.split("_").str[0])

# ============================================================
# 4. RESOLVE REPEATED AZD2014 MEASUREMENTS
# ============================================================
# Some cell lines have multiple AZD2014 AUC measurements because they were
# tested in different PRISM screens (MTS010 and HTS002) and/or in both
# PR300 and PR500 panels.
#
# Goal: retain exactly one AUC measurement per cell line using the
# predefined PRISM measurement hierarchy.

# Extract the PRISM panel (PR300 or PR500) from row_name.
azd2014["panel"] = azd2014["row_name"].str.split("_").str[0]

# Inspect the combinations of screen and panel present in the data.
pd.crosstab(azd2014["screen_id"], azd2014["panel"])

# Assign a priority to each measurement:
#   1 = MTS010 + PR500  -> preferred
#   2 = MTS010 + PR300  -> use if PR500 is unavailable
#   3 = HTS002 + PR500  -> use if MTS010 is unavailable
azd2014["priority"] = np.select(
    [
        (azd2014["screen_id"] == "MTS010") & (azd2014["panel"] == "PR500"),
        (azd2014["screen_id"] == "MTS010") & (azd2014["panel"] == "PR300"),
        (azd2014["screen_id"] == "HTS002") & (azd2014["panel"] == "PR500"),
    ],[1, 2, 3],default=np.nan)

# Verify that every measurement was assigned to one of the expected
# screen/panel combinations. There should be no NaN priorities.
azd2014["priority"].value_counts(dropna=False).sort_index()

# Sort measurements from highest to lowest preference and then keep
# the first measurement for each cell line.
#
# Example for a cell line measured three times:
#   MTS010 + PR500  -> priority 1 -> KEEP
#   MTS010 + PR300  -> priority 2 -> discard
#   HTS002 + PR500  -> priority 3 -> discard
azd2014_filtered = (azd2014.sort_values("priority").drop_duplicates(subset="depmap_id", keep="first").copy())

# ============================================================
# 5. VERIFY FINAL AZD2014 RESPONSE DATA
# ============================================================
# The filtered dataset should contain exactly one AUC measurement
# for each cell line.
print("Rows:", len(azd2014_filtered))
print("Unique cell lines:", azd2014_filtered["depmap_id"].nunique())

# Verify that no duplicated cell-line IDs remain.
print("Duplicated cell lines:",azd2014_filtered["depmap_id"].duplicated().sum())

# Examine which measurement source was ultimately selected.
azd2014_filtered["priority"].value_counts().sort_index()

# Final result:
#   720 cell lines with one AUC measurement each
#   473 from MTS010 + PR500 (priority 1)
#   237 from MTS010 + PR300 (priority 2)
#    10 from HTS002 + PR500 (priority 3)
#
# Most HTS002 measurements and some PR300 measurements were discarded
# because a higher-priority measurement was available for the same cell line.


# ============================================================
# 6. PREPROCESS RNA-SEQ DATA
# ============================================================
# The AZD2014 response dataset now contains 720 unique cell lines.
# Restrict the RNA-seq dataset to these cell lines before further
# preprocessing, since only these samples will be used for modeling.

azd2014_cell_lines = set(azd2014_filtered["depmap_id"])
expression_azd2014 = expression[expression["ModelID"].isin(azd2014_cell_lines)].copy()
print("RNA-seq profiles:", len(expression_azd2014))
print("Unique cell lines:", expression_azd2014["ModelID"].nunique())

# Identify AZD2014 cell lines that have multiple RNA-seq profiles.
expression_duplicates = expression_azd2014[expression_azd2014["ModelID"].duplicated(keep=False)].copy()
print("Cell lines with multiple RNA-seq profiles:",expression_duplicates["ModelID"].nunique())
# How many profiles does each duplicated cell line have?
expression_duplicates["ModelID"].value_counts().value_counts().sort_index()

expression_duplicates[["ModelID","ModelConditionID","SequencingID","IsDefaultEntryForMC"]].sort_values("ModelID").head(30)
expression_duplicates["IsDefaultEntryForMC"].value_counts(dropna=False)


# Verify that each duplicated cell line has exactly one profile
# marked as the default entry.
default_per_model = (expression_duplicates[expression_duplicates["IsDefaultEntryForMC"] == "Yes"].groupby("ModelID").size())
default_per_model.value_counts()

# ============================================================
# 7. SELECT ONE RNA-SEQ PROFILE PER CELL LINE
# ============================================================
# Among the 720 AZD2014 cell lines, 23 have multiple RNA-seq profiles.
# Each of these duplicated cell lines has exactly one profile marked
# as the default entry (IsDefaultEntryForMC == "Yes").
#
# Therefore, retain only the default RNA-seq profile for each cell line.
expression_filtered = expression_azd2014[expression_azd2014["IsDefaultEntryForMC"] == "Yes"].copy()

# Verify that the final expression dataset contains exactly one RNA-seq profile per cell line.
print("RNA-seq profiles:", len(expression_filtered))
print("Unique cell lines:", expression_filtered["ModelID"].nunique())
print("Duplicated cell lines:", expression_filtered["ModelID"].duplicated().sum())


# Verify that the filtered drug-response and RNA-seq datasets contain exactly the same set of cell lines.
azd_ids = set(azd2014_filtered["depmap_id"])
expression_ids = set(expression_filtered["ModelID"])

print("AZD2014 only:", len(azd_ids - expression_ids))
print("RNA-seq only:", len(expression_ids - azd_ids))

# ============================================================
# 8. MERGE RNA-SEQ EXPRESSION WITH AZD2014 RESPONSE
# ============================================================
# Remove RNA-seq metadata columns that are no longer needed.
# Keep ModelID as the unique cell-line identifier.
metadata_columns = ["Unnamed: 0", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]

expression_modeling = expression_filtered.drop(columns=metadata_columns).copy()

# Keep only the cell-line identifier and selected AUC target
# from the AZD2014 response data.
azd2014_target = azd2014_filtered[["depmap_id", "auc"]].copy()

# Rename depmap_id to ModelID so both datasets use the same
# identifier and we do not create two ID columns after merging.
azd2014_target = azd2014_target.rename(columns={"depmap_id": "ModelID"})

# Merge RNA-seq expression with the AZD2014 response.
modeling_data = expression_modeling.merge(azd2014_target, on="ModelID", how="inner")

# Verify the final merged dataset.
print("Rows:", len(modeling_data))
print("Unique cell lines:", modeling_data["ModelID"].nunique())
print("Missing AUC:", modeling_data["auc"].isna().sum())
print("Shape:", modeling_data.shape)

modeling_data.head()

# ============================================================
# 9. FINAL QC AND EXPLORATORY ANALYSIS OF MODELING DATA
# ============================================================
# Inspect the AZD2014 AUC target distribution.
# Lower AUC indicates greater drug sensitivity, whereas higher AUC
# indicates greater resistance.
#
# The target contains 720 observations, with AUC values ranging from
# approximately 0.42 to 0.94 (mean ~0.67). The distribution is roughly
# unimodal without obvious extreme outliers.

modeling_data["auc"].describe()
sns.displot(data=modeling_data, x='auc')

# let's inspect if some gene value is missing
gene_columns = modeling_data.columns.drop(["ModelID", "auc"])
print("Number of genes:", len(gene_columns))
print("Missing gene-expression values:", modeling_data[gene_columns].isna().sum().sum())

# Check for genes with zero variance across the 720 cell lines.
# Zero-variance genes have identical expression in every sample and
# therefore cannot contribute to predicting differences in drug response.
gene_variance = modeling_data[gene_columns].var()
zero_variance_genes = gene_variance[gene_variance == 0]
print("Zero-variance genes:", len(zero_variance_genes))  # 9

# Remove the 9 non-informative genes.
modeling_data = modeling_data.drop(columns=zero_variance_genes.index)
print("Shape after removing zero-variance genes:", modeling_data.shape)
# Final shape: 720 samples x 19,208 columns
# (19,206 genes + ModelID + AUC)

# ============================================================
# 10. SAVE FINAL MODELING DATASET
# ============================================================

# Save the cleaned dataset for use in the modeling script.
#
# Final dataset:
#   - 720 cell lines
#   - 19,206 gene-expression features
#   - ModelID: cell-line identifier
#   - auc: AZD2014 drug-response target
#
# Each cell line has exactly one RNA-seq profile and one AUC value.

modeling_data.to_csv(r"data\azd2014_modeling_data.csv",index=False)

print("Saved modeling dataset:", modeling_data.shape)