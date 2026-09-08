import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from drug_screening import screen_drug


# ============================================================
# MULTI-DRUG SCREENING SCRIPT
# ============================================================
#
# Purpose:
# Screen multiple drugs to identify drug-response phenotypes that can
# potentially be predicted from baseline RNA-seq gene-expression data.
#
# This script uses the reusable screen_drug() function defined in
# drug_screening.py. The detailed preprocessing, model fitting,
# evaluation, plotting, and result saving are handled by that function.
#
#
# DRUG SELECTION
# --------------
# The drugs tested in this script are selected from compounds with good
# overlap between the two required datasets:
#
#       1. RNA-seq gene-expression data
#       2. PRISM drug-response AUC data
#
# In other words, priority is given to drugs with a relatively large
# number of cell lines for which BOTH baseline RNA-seq measurements and
# drug-response measurements are available.
#
# Good overlap is important because the gene-expression dataset contains
# a very large number of predictors (~19,000 genes), while the number of
# available cell lines is much smaller. Maximizing the number of matched
# cell lines therefore gives the models as much training information as
# possible.
#
#
# SCREENING STRATEGY
# ------------------
# Each selected drug is passed independently to screen_drug().
#
# The same preprocessing procedure, train/test split strategy,
# cross-validation procedure, hyperparameter grids, and evaluation
# metrics are used for every drug. This creates a standardized comparison
# between drugs rather than changing the modeling procedure for each one.
#
# Models can be enabled or disabled depending on the desired screening
# speed:
#
#       enet=True/False  -> Elastic Net regression
#       svr=True/False   -> RBF Support Vector Regression
#       rf=True/False    -> Random Forest regression
#
# For rapid initial screening, computationally expensive models can be
# disabled and added later for drugs that appear more promising.
#
#
# GOAL
# ----
# The goal is NOT to optimize every drug extensively at this stage.
#
# Instead, the goal is to determine whether predictability from baseline
# gene expression differs substantially between drugs.
#
# For each drug, model performance is compared with a mean-only baseline
# using metrics such as:
#
#       - Cross-validation RMSE
#       - Test RMSE
#       - Test R²
#       - Train R²
#       - Improvement over the mean-only baseline
#
# Drugs showing substantially better and more consistent predictive
# performance can then be selected for deeper analysis, including more
# extensive modeling, feature selection, dimensionality reduction,
# biological interpretation, and/or integration of additional molecular
# features.
#
#
# RESULTS
# -------
# screen_drug() automatically saves the output for every tested drug in:
#
# drug_screening_function_results/
#     <drug_name>/
#
# Therefore, this script can be used to run many drugs sequentially while
# keeping the results of each drug organized separately for later
# comparison.
#
#
# Example:
#
# results_azd2014 = screen_drug(
#     expression,
#     drug_response,
#     "AZD2014",
#     enet=True,
#     svr=True,
#     rf=False,
#     show_plots=False
# )
#
# results_osimertinib = screen_drug(
#     expression,
#     drug_response,
#     "osimertinib",
#     enet=True,
#     svr=True,
#     rf=False,
#     show_plots=False
# )
#
#
# IMPORTANT:
# This stage is exploratory drug screening. The purpose is to identify
# promising drug-response prediction problems using a consistent
# pipeline, rather than to treat the screening results as final model
# performance estimates.
# ============================================================

# 1. LOAD DATA
expression = pd.read_csv( r"data\OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
drug_response = pd.read_csv( r"data\prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")
drug_counts_overlap = pd.read_csv(r"G:\My Drive\Shimon\stuffs\Data_Science_New_way\Portfolio\Self_projects\Project_3_drug_response_cancer\data\data_I created\drug_counts_overlap.csv")

# ============================================================
# AZD2014
# ============================================================

# AZD2014 (vistusertib) is an experimental anti-cancer drug that inhibits mTOR,
# specifically both the mTORC1 and mTORC2 signaling complexes. The mTOR pathway
# regulates important cellular processes including growth, proliferation, metabolism,
# and survival, and is frequently dysregulated in cancer.
# AZD2014 has been investigated in clinical trials for several cancer types.
# Therefore, variation in gene expression across cell lines may contain molecular
# features associated with sensitivity or resistance to AZD2014.


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


# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "AZD2014", enet=True, svr=True, rf=False)




# ============================================================
# cinacalcet
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "cinacalcet", enet=True, svr=True, rf=False)





# ============================================================
# osimertinib
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "osimertinib", enet=True, svr=True, rf=False)




# ============================================================
# teniposide
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "teniposide", enet=True, svr=True, rf=False)




# ============================================================
# floxuridine
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "floxuridine", enet=True, svr=True, rf=False)



# ============================================================
# tanesimycin
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "tanesimycin", enet=True, svr=True, rf=False)



# ============================================================
# ponatinib
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "ponatinib", enet=True, svr=True, rf=False)


# ============================================================
# BMS-754807
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "BMS-754807", enet=True, svr=True, rf=False)


# ============================================================
# pevonedistat
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "pevonedistat", enet=True, svr=True, rf=False)


# ============================================================
# talazoparib
# ============================================================
# 2. RUN PREPROCESSING AND MODELS
results = screen_drug(expression, drug_response, "talazoparib", enet=True, svr=True, rf=False)