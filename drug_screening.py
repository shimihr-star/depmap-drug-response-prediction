import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)



# ============================================================
# DRUG SCREENING FUNCTION
# ============================================================
#
# Purpose:
# Rapidly evaluate whether the response to a selected drug can be
# predicted from baseline RNA-seq gene-expression data.
#
# The function provides a standardized screening pipeline that can be
# applied to different drugs in the PRISM dataset. Its main purpose is
# to identify drugs that show sufficient predictive signal to justify
# more detailed modeling and biological investigation.
#
#
# DATA PREPARATION
# ----------------
# For the selected drug, the function:
#
# 1. Extracts all available PRISM drug-response measurements.
#
# 2. Keeps only cell lines that also have RNA-seq expression data.
#
# 3. Resolves repeated PRISM measurements using the predefined priority:
#       1. MTS010 + PR500
#       2. MTS010 + PR300
#       3. HTS002 + PR500
#
# 4. Keeps one drug-response measurement per cell line.
#
# 5. Selects the default RNA-seq profile for each cell line.
#
# 6. Merges gene-expression data with the drug-response AUC.
#
# 7. Removes RNA-seq metadata columns and zero-variance genes.
#
#
# DRUG-RESPONSE EXPLORATION
# -------------------------
# The function reports:
#       - Number of original PRISM measurements
#       - Number of matched cell lines
#       - Number of genes
#       - Number of zero-variance genes removed
#       - PRISM measurement-priority counts
#       - AUC mean, standard deviation, variance
#       - AUC quartiles, median, minimum and maximum
#
# It also generates a histogram of the AUC distribution.
#
#
# MODELING
# --------
# The data are divided into training and held-out test sets.
#
# A mean-only DummyRegressor is fitted as a baseline. This provides a
# reference for determining whether gene-expression-based models actually
# improve prediction over simply predicting the average drug response.
#
# The following models can then be independently enabled or disabled:
#
#       enet=True/False  -> Elastic Net regression
#       svr=True/False   -> RBF Support Vector Regression
#       rf=True/False    -> Random Forest regression
#
# Elastic Net and SVR use StandardScaler inside sklearn Pipelines so
# scaling is learned only from the appropriate training data.
#
# Hyperparameters are selected using GridSearchCV with cross-validation
# on the training set. Model selection minimizes RMSE.
#
#
# MODEL EVALUATION
# ----------------
# Each selected model is evaluated using:
#
#       MAE                    - Mean Absolute Error
#       RMSE                   - Root Mean Squared Error
#       R2                     - Test-set R²
#       CV_RMSE                - Best cross-validation RMSE
#       Train_R2               - Training-set R²
#       RMSE_over_test_SD      - RMSE relative to variation in test AUC
#       Improvement_vs_mean_%  - RMSE improvement over the mean baseline
#
# Train R² and test R² can also help reveal strong overfitting.
#
# For every model, the function generates:
#       - Observed vs predicted AUC plot
#       - Residuals vs predicted AUC plot
#
#
# SAVING RESULTS
# --------------
# Results are automatically stored in:
#
# drug_screening_function_results/
#     <drug_name>/
#
# The main drug folder contains:
#       - auc_distribution.png
#       - data_summary.csv
#       - model_results.csv
#       - test_set_summary.csv
#
# Each fitted model receives its own subfolder containing:
#       - results.txt
#       - predictions.csv
#       - observed_vs_predicted.png
#       - residuals_vs_predicted.png
#
# Setting show_plots=False prevents plots from being displayed during
# execution, but the plots are still generated and saved to disk.
#
#
# OUTPUT
# ------
# The function returns a dictionary containing:
#       - Drug name
#       - QC information
#       - AUC distribution statistics
#       - Model-performance table
#       - Fitted models
#       - Model predictions
#       - Processed modeling dataset
#       - Training and test datasets
#       - Path to the saved results folder
#
#
# Example:
#
# results = screen_drug(
#     expression,
#     drug_response,
#     "AZD2014",
#     enet=True,
#     svr=True,
#     rf=False,
#     show_plots=False
# )
#
#
# IMPORTANT:
# This function is intended for rapid and standardized drug screening,
# not as the final modeling pipeline. Its purpose is to compare the
# predictability of different drug responses from baseline gene
# expression and identify promising drugs for deeper analysis.
# ============================================================



# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_drug_data(expression, drug_response, drug_name):
    """
    Prepare RNA-seq + PRISM drug-response data for one selected drug.

    PRISM measurement priority:
        1. MTS010 + PR500
        2. MTS010 + PR300
        3. HTS002 + PR500
    """

    required_exp = {
        "ModelID",
        "IsDefaultEntryForMC"
    }

    required_dr = {
        "name",
        "depmap_id",
        "auc",
        "screen_id",
        "row_name"
    }

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    if not required_exp.issubset(expression.columns):

        missing = (
            required_exp
            - set(expression.columns)
        )

        raise ValueError(
            f"Missing expression columns: {missing}"
        )

    if not required_dr.issubset(drug_response.columns):

        missing = (
            required_dr
            - set(drug_response.columns)
        )

        raise ValueError(
            f"Missing drug-response columns: {missing}"
        )

    # --------------------------------------------------------
    # Select drug
    # --------------------------------------------------------

    dr = drug_response[
        drug_response["name"] == drug_name
    ].copy()

    if dr.empty:

        raise ValueError(
            f"No PRISM rows found for drug: {drug_name}"
        )

    raw_n = len(dr)

    # --------------------------------------------------------
    # Keep only cell lines with RNA-seq data
    # --------------------------------------------------------

    expression_ids = set(
        expression["ModelID"].dropna()
    )

    dr = dr[
        dr["depmap_id"].isin(
            expression_ids
        )
    ].copy()

    # --------------------------------------------------------
    # Extract PRISM panel
    # --------------------------------------------------------

    dr["panel"] = (
        dr["row_name"]
        .astype(str)
        .str.split("_")
        .str[0]
    )

    # --------------------------------------------------------
    # PRISM measurement hierarchy
    # --------------------------------------------------------

    dr["priority"] = np.select(
        [
            (
                (dr["screen_id"] == "MTS010")
                & (dr["panel"] == "PR500")
            ),

            (
                (dr["screen_id"] == "MTS010")
                & (dr["panel"] == "PR300")
            ),

            (
                (dr["screen_id"] == "HTS002")
                & (dr["panel"] == "PR500")
            )
        ],

        [1, 2, 3],

        default=np.nan
    )

    unknown = int(
        dr["priority"]
        .isna()
        .sum()
    )

    if unknown > 0:

        warnings.warn(
            f"Dropping {unknown} measurements outside "
            f"the predefined PRISM hierarchy."
        )

        dr = dr.dropna(
            subset=["priority"]
        )

    if dr.empty:

        raise ValueError(
            f"No usable measurements remain for "
            f"{drug_name} after applying the "
            f"PRISM hierarchy."
        )

    # --------------------------------------------------------
    # Keep one drug-response measurement per cell line
    # --------------------------------------------------------

    dr = (
        dr
        .sort_values("priority")
        .drop_duplicates(
            subset="depmap_id",
            keep="first"
        )
        .copy()
    )

    # --------------------------------------------------------
    # Select corresponding RNA-seq samples
    # --------------------------------------------------------

    ids = set(
        dr["depmap_id"]
    )

    exp = expression[
        expression["ModelID"].isin(ids)
    ].copy()

    # --------------------------------------------------------
    # Keep default RNA-seq profile
    # --------------------------------------------------------

    exp = exp[
        exp["IsDefaultEntryForMC"] == "Yes"
    ].copy()

    # Defensive check
    if exp["ModelID"].duplicated().any():

        warnings.warn(
            "Duplicate default RNA-seq profiles found. "
            "Keeping first profile per ModelID."
        )

        exp = exp.drop_duplicates(
            subset="ModelID",
            keep="first"
        )

    # --------------------------------------------------------
    # Remove RNA-seq metadata
    # --------------------------------------------------------

    metadata_columns = [
        "Unnamed: 0",
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel"
    ]

    columns_to_drop = [
        c
        for c in metadata_columns
        if c in exp.columns
    ]

    exp = exp.drop(
        columns=columns_to_drop
    )

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    target = (
        dr[["depmap_id", "auc"]]
        .rename(
            columns={
                "depmap_id": "ModelID"
            }
        )
        .copy()
    )

    # --------------------------------------------------------
    # Merge RNA-seq and response
    # --------------------------------------------------------

    data = exp.merge(
        target,
        on="ModelID",
        how="inner"
    )

    data = data.dropna(
        subset=["auc"]
    )

    # --------------------------------------------------------
    # Gene-expression QC
    # --------------------------------------------------------

    gene_columns = data.columns.drop(
        ["ModelID", "auc"]
    )

    missing_gene_values = int(
        data[gene_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_gene_values > 0:

        raise ValueError(
            f"{missing_gene_values} missing "
            f"gene-expression values found."
        )

    # --------------------------------------------------------
    # Remove zero-variance genes
    # --------------------------------------------------------

    gene_variance = (
        data[gene_columns]
        .var()
    )

    zero_variance_genes = (
        gene_variance[
            gene_variance == 0
        ]
        .index
    )

    data = data.drop(
        columns=zero_variance_genes
    )

    # --------------------------------------------------------
    # QC summary
    # --------------------------------------------------------

    qc = {

        "raw_measurements":
            raw_n,

        "matched_cell_lines":
            data["ModelID"].nunique(),

        "genes":
            data.shape[1] - 2,

        "zero_variance_genes_removed":
            len(zero_variance_genes),

        "priority_counts":
            dr["priority"]
            .value_counts()
            .sort_index()
            .to_dict()
    }

    return data, qc


# ============================================================
# MODEL DIAGNOSTIC PLOTS
# ============================================================

def _diagnostic_plots(
        y_test,
        pred,
        drug_name,
        model_name,
        model_folder,
        show_plots=True):

    """
    Create and save diagnostic plots for one model.
    """

    residuals = (
        y_test.to_numpy()
        - np.asarray(pred)
    )

    # --------------------------------------------------------
    # Observed vs predicted
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        y_test,
        pred,
        alpha=0.7
    )

    low = min(
        y_test.min(),
        np.min(pred)
    )

    high = max(
        y_test.max(),
        np.max(pred)
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--"
    )

    ax.set_xlabel(
        "Observed AUC"
    )

    ax.set_ylabel(
        "Predicted AUC"
    )

    ax.set_title(
        f"{drug_name} - {model_name}\n"
        f"Observed vs Predicted"
    )

    fig.tight_layout()

    fig.savefig(
        model_folder
        / "observed_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    if show_plots:
        plt.show()

    plt.close(fig)

    # --------------------------------------------------------
    # Residuals vs predicted
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        pred,
        residuals,
        alpha=0.7
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Predicted AUC"
    )

    ax.set_ylabel(
        "Residual"
    )

    ax.set_title(
        f"{drug_name} - {model_name}\n"
        f"Residuals vs Predicted"
    )

    fig.tight_layout()

    fig.savefig(
        model_folder
        / "residuals_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    if show_plots:
        plt.show()

    plt.close(fig)


# ============================================================
# DRUG SCREENING FUNCTION
# ============================================================
#
# Purpose:
# Quickly evaluate whether the response to a selected drug can be
# predicted from baseline RNA-seq gene-expression data.
#
# This function is intended as a screening tool for identifying drugs
# that show enough predictive signal to justify deeper modeling and
# biological investigation.
#
# The function:
#
# 1. Selects the requested drug from the PRISM drug-response dataset.
#
# 2. Resolves repeated PRISM measurements for the same cell line using
#    the predefined measurement priority:
#       1. MTS010 + PR500
#       2. MTS010 + PR300
#       3. HTS002 + PR500
#
# 3. Matches drug-response measurements with available RNA-seq data.
#
# 4. Keeps one default RNA-seq profile per cell line.
#
# 5. Removes RNA-seq metadata columns and zero-variance genes.
#
# 6. Reports and plots the AUC distribution.
#
# 7. Splits the data into training and held-out test sets.
#
# 8. Fits a mean-only baseline model for comparison.
#
# 9. Optionally trains:
#       - Elastic Net
#       - RBF-SVR
#       - Random Forest
#
# 10. Hyperparameters are selected using cross-validation on the
#     training data, optimizing RMSE.
#
# 11. Evaluates each selected model using:
#       - MAE
#       - RMSE
#       - R2
#       - CV RMSE
#       - Train R2
#       - RMSE / test SD
#       - Improvement relative to mean-only baseline
#
# 12. Generates:
#       - Observed vs predicted plots
#       - Residual plots
#
# 13. Automatically saves all plots and numerical results inside:
#
#       drug_screening_function_results/
#           drug_name/
#
# IMPORTANT:
# This is a rapid screening tool, not the final modeling pipeline.
# Its purpose is to identify promising drugs before investing time
# in more extensive feature engineering and model development.
# ============================================================


def screen_drug(
        expression,
        drug_response,
        drug_name,
        rf=True,
        enet=True,
        svr=True,
        test_size=0.15,
        random_state=101,
        cv=5,
        show_plots=True,
        results_root="drug_screening_function_results"):

    # ========================================================
    # CREATE RESULTS FOLDER
    # ========================================================

    results_root = Path(
        results_root
    )

    drug_folder = (
        results_root
        / drug_name
    )

    drug_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    data, qc = prepare_drug_data(
        expression,
        drug_response,
        drug_name
    )

    if len(data) < 50:

        warnings.warn(
            f"Only {len(data)} matched cell lines. "
            f"Model estimates may be unstable."
        )

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    modeling_data = (
        data.set_index("ModelID")
    )

    X = modeling_data.drop(
        columns="auc"
    )

    y = modeling_data["auc"]

    # ========================================================
    # AUC DISTRIBUTION
    # ========================================================

    distribution = (
        y.describe()
    )

    print(
        f"\n{'=' * 65}"
        f"\n{drug_name} - DATA"
        f"\n{'=' * 65}"
    )

    print(
        pd.Series(qc)
        .to_string()
    )

    print(
        "\nAUC distribution:"
    )

    print(
        distribution.to_string()
    )

    print(
        f"variance    "
        f"{y.var():.6f}"
    )

    # ========================================================
    # SAVE DATA SUMMARY
    # ========================================================

    summary = pd.Series({

        "drug":
            drug_name,

        "n_cell_lines":
            len(y),

        "n_genes":
            X.shape[1],

        "auc_mean":
            y.mean(),

        "auc_std":
            y.std(),

        "auc_variance":
            y.var(),

        "auc_min":
            y.min(),

        "auc_25%":
            y.quantile(0.25),

        "auc_median":
            y.median(),

        "auc_75%":
            y.quantile(0.75),

        "auc_max":
            y.max()
    })

    summary.to_csv(
        drug_folder
        / "data_summary.csv",
        header=["value"]
    )

    # ========================================================
    # AUC DISTRIBUTION PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.hist(
        y,
        bins=25,
        edgecolor="black",
        alpha=0.75
    )

    ax.axvline(
        y.mean(),
        linestyle="--",
        label=(
            f"Mean = "
            f"{y.mean():.3f}"
        )
    )

    ax.axvline(
        y.median(),
        linestyle=":",
        label=(
            f"Median = "
            f"{y.median():.3f}"
        )
    )

    ax.set_xlabel(
        "AUC"
    )

    ax.set_ylabel(
        "Count"
    )

    ax.set_title(
        f"{drug_name}: "
        f"AUC distribution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        drug_folder
        / "auc_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    if show_plots:
        plt.show()

    plt.close(fig)

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # ========================================================
    # DEFINE MODELS
    # ========================================================

    specs = {}

    # --------------------------------------------------------
    # Elastic Net
    # --------------------------------------------------------

    if enet:

        specs["Elastic Net"] = (

            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "elastic_net",
                    ElasticNet(
                        max_iter=50000,
                        tol=1e-4
                    )
                )
            ]),

            {
                "elastic_net__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1,
                    10
                ],

                "elastic_net__l1_ratio": [
                    0.01,
                    0.05,
                    0.1,
                    0.3,
                    0.5,
                    0.7,
                    1
                ]
            }
        )

    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    if svr:

        specs["SVR"] = (

            Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "svr",
                    SVR(
                        kernel="rbf"
                    )
                )
            ]),

            {
                "svr__C": [
                    0.1,
                    1,
                    10,
                    100
                ],

                "svr__gamma": [
                    "scale",
                    0.001,
                    0.01,
                    0.1
                ],

                "svr__epsilon": [
                    0.01,
                    0.05,
                    0.1
                ]
            }
        )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if rf:

        specs["Random Forest"] = (

            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),

            {
                "n_estimators": [
                    300,
                    500
                ],

                "max_features": [
                    "sqrt",
                    0.1,
                    0.3
                ],

                "max_depth": [
                    None,
                    10,
                    20
                ],

                "min_samples_leaf": [
                    1,
                    3,
                    5
                ]
            }
        )

    # --------------------------------------------------------
    # At least one model must be selected
    # --------------------------------------------------------

    if not specs:

        raise ValueError(
            "At least one model must be selected: "
            "enet=True, svr=True, or rf=True."
        )

    # ========================================================
    # MEAN-ONLY BASELINE
    # ========================================================

    dummy = DummyRegressor(
        strategy="mean"
    )

    dummy.fit(
        X_train,
        y_train
    )

    dummy_pred = dummy.predict(
        X_test
    )

    rows = [

        {
            "model":
                "Mean baseline",

            "MAE":
                mean_absolute_error(
                    y_test,
                    dummy_pred
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        dummy_pred
                    )
                ),

            "R2":
                r2_score(
                    y_test,
                    dummy_pred
                ),

            "CV_RMSE":
                np.nan,

            "Train_R2":
                np.nan,

            "best_params":
                {}
        }
    ]

    fitted_models = {
        "Mean baseline":
            dummy
    }

    predictions = {
        "Mean baseline":
            dummy_pred
    }

    # ========================================================
    # FIT SELECTED MODELS
    # ========================================================

    for model_name, (
            estimator,
            param_grid
    ) in specs.items():

        print(
            f"\nFitting "
            f"{model_name}..."
        )

        # ----------------------------------------------------
        # Create folder for this model
        # ----------------------------------------------------

        safe_model_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        model_folder = (
            drug_folder
            / safe_model_name
        )

        model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Grid search
        # ----------------------------------------------------

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=(
                "neg_root_mean_squared_error"
            ),
            n_jobs=-1
        )

        search.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        pred = search.predict(
            X_test
        )

        train_pred = search.predict(
            X_train
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        r2 = r2_score(
            y_test,
            pred
        )

        train_r2 = r2_score(
            y_train,
            train_pred
        )

        cv_rmse = (
            -search.best_score_
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        rows.append({

            "model":
                model_name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,

            "CV_RMSE":
                cv_rmse,

            "Train_R2":
                train_r2,

            "best_params":
                search.best_params_
        })

        fitted_models[
            model_name
        ] = search

        predictions[
            model_name
        ] = pred

        # ====================================================
        # SAVE MODEL RESULTS
        # ====================================================

        with open(
            model_folder
            / "results.txt",
            "w"
        ) as f:

            f.write(
                f"Drug: "
                f"{drug_name}\n"
            )

            f.write(
                f"Model: "
                f"{model_name}\n\n"
            )

            f.write(
                f"MAE: "
                f"{mae:.6f}\n"
            )

            f.write(
                f"RMSE: "
                f"{rmse:.6f}\n"
            )

            f.write(
                f"R2: "
                f"{r2:.6f}\n"
            )

            f.write(
                f"CV RMSE: "
                f"{cv_rmse:.6f}\n"
            )

            f.write(
                f"Train R2: "
                f"{train_r2:.6f}\n\n"
            )

            f.write(
                "Best parameters:\n"
            )

            f.write(
                str(
                    search.best_params_
                )
            )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        prediction_df = pd.DataFrame({

            "observed_auc":
                y_test,

            "predicted_auc":
                pred,

            "residual":
                y_test.to_numpy()
                - pred
        })

        prediction_df.to_csv(
            model_folder
            / "predictions.csv"
        )

        # ----------------------------------------------------
        # Diagnostic plots
        # ----------------------------------------------------

        _diagnostic_plots(
            y_test,
            pred,
            drug_name,
            model_name,
            model_folder,
            show_plots=show_plots
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = (
        pd.DataFrame(rows)
        .set_index("model")
    )

    # --------------------------------------------------------
    # RMSE relative to test SD
    # --------------------------------------------------------

    test_sd = (
        y_test.std()
    )

    results_df[
        "RMSE_over_test_SD"
    ] = (
        results_df["RMSE"]
        / test_sd
    )

    # --------------------------------------------------------
    # Improvement over mean baseline
    # --------------------------------------------------------

    baseline_rmse = (
        results_df.loc[
            "Mean baseline",
            "RMSE"
        ]
    )

    results_df[
        "Improvement_vs_mean_%"
    ] = (

        (
            baseline_rmse
            - results_df["RMSE"]
        )

        / baseline_rmse

        * 100
    )

    # ========================================================
    # SAVE OVERALL MODEL RESULTS
    # ========================================================

    results_df.to_csv(
        drug_folder
        / "model_results.csv"
    )

    # ========================================================
    # SAVE TEST SET INFORMATION
    # ========================================================

    test_set_summary = pd.Series({

        "n_train":
            len(y_train),

        "n_test":
            len(y_test),

        "test_size":
            test_size,

        "random_state":
            random_state,

        "cv_folds":
            cv,

        "test_auc_mean":
            y_test.mean(),

        "test_auc_std":
            y_test.std()
    })

    test_set_summary.to_csv(
        drug_folder
        / "test_set_summary.csv",
        header=["value"]
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        f"\n{'=' * 65}"
        f"\n{drug_name} - MODEL SCREEN"
        f"\n{'=' * 65}"
    )

    display_columns = [

        "MAE",
        "RMSE",
        "R2",
        "CV_RMSE",
        "Train_R2",
        "RMSE_over_test_SD",
        "Improvement_vs_mean_%"
    ]

    print(
        results_df[
            display_columns
        ]
        .round(4)
        .to_string()
    )

    # --------------------------------------------------------
    # Best hyperparameters
    # --------------------------------------------------------

    print(
        "\nBest parameters:"
    )

    for model_name in specs:

        print(
            f"{model_name}: "
            f"{results_df.loc[model_name, 'best_params']}"
        )

    print(
        f"\nResults saved to:"
        f"\n{drug_folder.resolve()}"
    )

    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return {

        "drug":
            drug_name,

        "qc":
            qc,

        "distribution":
            distribution,

        "model_results":
            results_df,

        "models":
            fitted_models,

        "predictions":
            predictions,

        "modeling_data":
            data,

        "X_train":
            X_train,

        "X_test":
            X_test,

        "y_train":
            y_train,

        "y_test":
            y_test,

        "results_folder":
            drug_folder
    }