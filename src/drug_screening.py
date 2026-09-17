import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = (PROJECT_ROOT / "results" / "drug_screening")


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_drug_data(expression, drug_response, drug_name,):
    """
    Prepare RNA-seq and PRISM drug-response data for one drug.
    PRISM measurement priority:
        1. MTS010 + PR500
        2. MTS010 + PR300
        3. HTS002 + PR500

    Returns
    -------
    data : pandas.DataFrame
        One row per biological cell line containing ModelID,
        gene-expression features, and drug-response AUC.

    qc : dict
        Data-preparation quality-control information.
    """

    required_expression_columns = {"ModelID", "IsDefaultEntryForMC",}
    required_drug_columns = {"name", "depmap_id", "auc", "screen_id", "row_name",}

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_expression_columns = (required_expression_columns - set(expression.columns))

    if missing_expression_columns:
        raise ValueError("Missing expression columns: " f"{missing_expression_columns}")

    missing_drug_columns = (required_drug_columns - set(drug_response.columns))

    if missing_drug_columns:
        raise ValueError("Missing drug-response columns: "f"{missing_drug_columns}")

    # --------------------------------------------------------
    # Select requested drug
    # --------------------------------------------------------

    dr = drug_response[drug_response["name"] == drug_name].copy()

    if dr.empty:
        raise ValueError(f"No PRISM rows found for drug: {drug_name}")

    raw_measurements = len(dr)

    # --------------------------------------------------------
    # Keep cell lines with RNA-seq data
    # --------------------------------------------------------

    expression_ids = set(expression["ModelID"].dropna())
    dr = dr[dr["depmap_id"].isin(expression_ids)].copy()
    matched_before_priority = (dr["depmap_id"].nunique())

    if dr.empty:
        raise ValueError( f"No RNA-seq/PRISM overlap found for {drug_name}.")

    # --------------------------------------------------------
    # Extract PRISM panel
    # --------------------------------------------------------

    dr["panel"] = (dr["row_name"].astype(str).str.split("_").str[0])

    # --------------------------------------------------------
    # Assign PRISM measurement priority
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
            ),
        ],
        [1, 2, 3],
        default=np.nan,
    )

    unknown_priority = int(dr["priority"].isna().sum())

    if unknown_priority > 0:
        warnings.warn(
            f"Dropping {unknown_priority} {drug_name} measurements "
            "outside the predefined PRISM hierarchy.")

        dr = dr.dropna(subset=["priority"])

    if dr.empty:
        raise ValueError(
            f"No usable measurements remain for {drug_name} "
            "after applying the PRISM hierarchy.")

    # --------------------------------------------------------
    # Keep one drug-response measurement per cell line
    # --------------------------------------------------------

    dr = (dr.sort_values("priority").drop_duplicates(subset="depmap_id", keep="first",).copy())

    # --------------------------------------------------------
    # Select corresponding RNA-seq profiles
    # --------------------------------------------------------

    selected_ids = set(dr["depmap_id"])
    exp = expression[expression["ModelID"].isin(selected_ids)].copy()

    # --------------------------------------------------------
    # Keep default RNA-seq profile
    # --------------------------------------------------------

    exp = exp[exp["IsDefaultEntryForMC"] == "Yes"].copy()

    if exp["ModelID"].duplicated().any():
        duplicate_count = (exp.loc[exp["ModelID"].duplicated(keep=False),"ModelID",].nunique())

        warnings.warn(
            f"{duplicate_count} cell lines have multiple default "
            "RNA-seq profiles. Keeping the first profile per ModelID.")

        exp = exp.drop_duplicates(
            subset="ModelID",
            keep="first",)

    # --------------------------------------------------------
    # Remove RNA-seq metadata
    # --------------------------------------------------------

    metadata_columns = ["Unnamed: 0", "SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel",]

    columns_to_drop = [
        column
        for column in metadata_columns
        if column in exp.columns]

    exp = exp.drop(columns=columns_to_drop)

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    target = (dr[["depmap_id", "auc"]].rename( columns={"depmap_id": "ModelID",}).copy())

    # --------------------------------------------------------
    # Merge expression and response
    # --------------------------------------------------------

    data = exp.merge(target, on="ModelID", how="inner",)

    data = data.dropna(subset=["auc"])

    if data.empty:
        raise ValueError(f"No complete modeling observations remain for {drug_name}.")

    # --------------------------------------------------------
    # Gene-expression QC
    # --------------------------------------------------------

    gene_columns = data.columns.drop(["ModelID", "auc"])

    missing_gene_values = int(data[gene_columns].isna().sum().sum())

    if missing_gene_values > 0:
        raise ValueError(
            f"{missing_gene_values} missing gene-expression "
            f"values found for {drug_name}.")

    # --------------------------------------------------------
    # Remove zero-variance genes
    # --------------------------------------------------------

    gene_variance = (data[gene_columns].var())
    zero_variance_genes = (gene_variance[gene_variance == 0].index)
    data = data.drop(columns=zero_variance_genes)

    # --------------------------------------------------------
    # QC summary
    # --------------------------------------------------------

    qc = {
        "raw_measurements": raw_measurements,
        "matched_before_priority": matched_before_priority,
        "matched_cell_lines": data["ModelID"].nunique(),
        "genes": data.shape[1] - 2,
        "zero_variance_genes_removed": len(
            zero_variance_genes
        ),
        "measurements_outside_priority": unknown_priority,
        "priority_1_count": int(
            (dr["priority"] == 1).sum()
        ),
        "priority_2_count": int(
            (dr["priority"] == 2).sum()
        ),
        "priority_3_count": int(
            (dr["priority"] == 3).sum()
        ),
    }

    return data, qc


# ============================================================
# MODEL DIAGNOSTIC PLOTS
# ============================================================

def _diagnostic_plots(y_test, predictions, drug_name, model_name, model_folder, show_plots=True,):
    """
    Create and save observed-vs-predicted and residual plots.
    """

    observed = y_test.to_numpy()
    predicted = np.asarray(predictions)
    residuals = observed - predicted

    # --------------------------------------------------------
    # Observed vs predicted
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(observed, predicted,alpha=0.7,)
    low = min(observed.min(),predicted.min(),)
    high = max(observed.max(), predicted.max(),)
    ax.plot([low, high],[low, high], linestyle="--",)
    ax.set_xlabel("Observed AUC")
    ax.set_ylabel("Predicted AUC")
    ax.set_title(f"{drug_name} - {model_name}\n" "Observed vs Predicted")
    fig.tight_layout()
    fig.savefig(model_folder / "observed_vs_predicted.png", dpi=300, bbox_inches="tight",)

    if show_plots:
        plt.show()

    plt.close(fig)

    # --------------------------------------------------------
    # Residuals vs predicted
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predicted, residuals, alpha=0.7,)
    ax.axhline(0, linestyle="--",)
    ax.set_xlabel("Predicted AUC")
    ax.set_ylabel("Residual")
    ax.set_title(f"{drug_name} - {model_name}\n" "Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(model_folder / "residuals_vs_predicted.png", dpi=300, bbox_inches="tight",)

    if show_plots:
        plt.show()

    plt.close(fig)


# ============================================================
# MAIN DRUG SCREENING FUNCTION
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
    results_root=None,
):
    """
    Run a standardized RNA-seq drug-response screening analysis.

    Parameters
    ----------
    expression : pandas.DataFrame
        DepMap RNA-seq expression dataset.

    drug_response : pandas.DataFrame
        PRISM drug-response dataset.

    drug_name : str
        Drug name exactly as represented in the PRISM dataset.

    rf : bool, default=True
        Fit Random Forest.

    enet : bool, default=True
        Fit Elastic Net.

    svr : bool, default=True
        Fit RBF Support Vector Regression.

    test_size : float, default=0.15
        Fraction of observations reserved for the held-out test set.

    random_state : int, default=101
        Random state used for the train/test split.

    cv : int, default=5
        Number of cross-validation folds used by GridSearchCV.

    show_plots : bool, default=True
        Display plots interactively. Plots are saved regardless.

    results_root : str or pathlib.Path, optional
        Root directory for screening results. If omitted, results
        are saved under results/drug_screening/.

    Returns
    -------
    dict
        Dictionary containing QC information, distribution statistics,
        model results, fitted models, predictions, processed data,
        train/test sets, and the output folder.
    """

    # ========================================================
    # VALIDATE ARGUMENTS
    # ========================================================

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if cv < 2:
        raise ValueError("cv must be at least 2.")

    if not any([enet, svr, rf]):
        raise ValueError("At least one model must be selected: " "enet=True, svr=True, or rf=True.")

    # ========================================================
    # CREATE RESULTS FOLDER
    # ========================================================

    if results_root is None:
        results_root = DEFAULT_RESULTS_ROOT

    results_root = Path(results_root)

    # Avoid problematic path characters if another compound name
    # contains characters such as "/" or "\".
    safe_drug_name = (str(drug_name).replace("/", "_").replace("\\", "_").replace(":", "_"))
    drug_folder = (results_root / safe_drug_name)
    drug_folder.mkdir(parents=True,  exist_ok=True,)

    # ========================================================
    # PREPARE DATA
    # ========================================================

    data, qc = prepare_drug_data(expression, drug_response, drug_name,)

    if len(data) < 50:
        warnings.warn(
            f"Only {len(data)} matched cell lines are available "
            f"for {drug_name}. Model estimates may be unstable."
        )

    modeling_data = (data.set_index("ModelID"))

    X = modeling_data.drop(columns="auc")
    y = modeling_data["auc"]

    if y.nunique() < 2:
        raise ValueError(
            f"{drug_name} has insufficient AUC variation for regression.")

    # ========================================================
    # AUC DISTRIBUTION
    # ========================================================

    distribution = y.describe()

    print(
        f"\n{'=' * 65}"
        f"\n{drug_name} - DATA"
        f"\n{'=' * 65}"
    )

    print(pd.Series(qc).to_string())
    print("\nAUC distribution:")
    print(distribution.to_string())

    print(f"variance    {y.var():.6f}")

    # ========================================================
    # SAVE DATA SUMMARY
    # ========================================================

    summary = pd.Series(
        {
            "drug": drug_name,
            "n_cell_lines": len(y),
            "n_genes": X.shape[1],
            "auc_mean": y.mean(),
            "auc_std": y.std(),
            "auc_variance": y.var(),
            "auc_min": y.min(),
            "auc_25%": y.quantile(0.25),
            "auc_median": y.median(),
            "auc_75%": y.quantile(0.75),
            "auc_max": y.max(),
        })

    summary.to_csv(drug_folder / "data_summary.csv", header=["value"],)

    # ========================================================
    # AUC DISTRIBUTION PLOT
    # ========================================================

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(y, bins=25, alpha=0.75,)
    ax.axvline(y.mean(), linestyle="--", label=f"Mean = {y.mean():.3f}",)
    ax.axvline(y.median(), linestyle=":", label=f"Median = {y.median():.3f}",)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"{drug_name}: AUC Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(drug_folder  / "auc_distribution.png", dpi=300, bbox_inches="tight",)

    if show_plots:
        plt.show()

    plt.close(fig)

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,)

    # ========================================================
    # MEAN-ONLY BASELINE
    # ========================================================

    dummy = DummyRegressor(strategy="mean")

    dummy.fit(X_train, y_train,)

    dummy_pred = dummy.predict( X_test)

    baseline_mae = mean_absolute_error(y_test,dummy_pred,)

    baseline_rmse = np.sqrt(mean_squared_error(y_test, dummy_pred,))

    baseline_r2 = r2_score(
        y_test,
        dummy_pred,
    )

    rows = [
        {
            "model": "Mean baseline",
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "R2": baseline_r2,
            "CV_RMSE": np.nan,
            "Train_R2": np.nan,
            "best_params": {},
        }
    ]

    fitted_models = {"Mean baseline": dummy}
    predictions = {"Mean baseline": dummy_pred}

    # ========================================================
    # DEFINE MODEL SPECIFICATIONS
    # ========================================================

    specs = {}

    if enet:
        specs["Elastic Net"] = (
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "elastic_net",
                        ElasticNet(
                            max_iter=50000,
                            tol=1e-4,
                        ),
                    ),
                ]
            ),
            {
                "elastic_net__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1,
                    10,
                ],
                "elastic_net__l1_ratio": [
                    0.01,
                    0.05,
                    0.1,
                    0.3,
                    0.5,
                    0.7,
                    1,
                ],
            },
        )

    if svr:
        specs["SVR"] = (
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "svr",
                        SVR(
                            kernel="rbf"
                        ),
                    ),
                ]
            ),
            {
                "svr__C": [
                    0.1,
                    1,
                    10,
                    100,
                ],
                "svr__gamma": [
                    "scale",
                    0.001,
                    0.01,
                    0.1,
                ],
                "svr__epsilon": [
                    0.01,
                    0.05,
                    0.1,
                ],
            },
        )

    if rf:
        specs["Random Forest"] = (
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [
                    300,
                    500,
                ],
                "max_features": [
                    "sqrt",
                    0.1,
                    0.3,
                ],
                "max_depth": [
                    None,
                    10,
                    20,
                ],
                "min_samples_leaf": [
                    1,
                    3,
                    5,
                ],
            },
        )

    # ========================================================
    # FIT SELECTED MODELS
    # ========================================================

    for model_name, (estimator, param_grid,) in specs.items():

        print(f"\nFitting {model_name}...")

        safe_model_name = (model_name.replace(" ", "_").replace("/", "_"))

        model_folder = (drug_folder  / safe_model_name)

        model_folder.mkdir(parents=True,exist_ok=True,)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,)

        search.fit(X_train, y_train,)
        pred = search.predict( X_test)
        train_pred = search.predict(X_train)
        mae = mean_absolute_error(y_test,pred,)

        rmse = np.sqrt(mean_squared_error( y_test,pred,))
        test_r2 = r2_score(y_test,pred,)
        train_r2 = r2_score(y_train,train_pred,)
        cv_rmse = (-search.best_score_)

        rows.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": test_r2,
                "CV_RMSE": cv_rmse,
                "Train_R2": train_r2,
                "best_params": search.best_params_,
            }
        )

        fitted_models[model_name] = search
        predictions[model_name] = pred

        # ----------------------------------------------------
        # Save text results
        # ----------------------------------------------------

        with open(model_folder / "results.txt", "w", encoding="utf-8",) as file:

            file.write(f"Drug: {drug_name}\n")
            file.write(f"Model: {model_name}\n\n")
            file.write(f"MAE: {mae:.6f}\n")
            file.write(f"RMSE: {rmse:.6f}\n")
            file.write(f"Test R2: {test_r2:.6f}\n")
            file.write(f"CV RMSE: {cv_rmse:.6f}\n")
            file.write(f"Train R2: {train_r2:.6f}\n\n")
            file.write("Best parameters:\n")
            file.write(str(search.best_params_))

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                "observed_auc": y_test,
                "predicted_auc": pred,
                "residual": (
                    y_test.to_numpy()
                    - pred
                ),}, index=y_test.index,)

        prediction_df.index.name = "ModelID"
        prediction_df.to_csv(model_folder / "predictions.csv")

        # ----------------------------------------------------
        # Diagnostic plots
        # ----------------------------------------------------

        _diagnostic_plots(
            y_test=y_test,
            predictions=pred,
            drug_name=drug_name,
            model_name=model_name,
            model_folder=model_folder,
            show_plots=show_plots,
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = (pd.DataFrame(rows).set_index("model"))
    test_sd = y_test.std()

    if test_sd > 0:
        results_df["RMSE_over_test_SD"] = (results_df["RMSE"] / test_sd)
    else:
        results_df["RMSE_over_test_SD"] = np.nan

    results_df["Improvement_vs_mean_%"] = ((baseline_rmse - results_df["RMSE"]) / baseline_rmse* 100)

    # ========================================================
    # SAVE OVERALL RESULTS
    # ========================================================

    results_df.to_csv(drug_folder / "model_results.csv")

    test_set_summary = pd.Series(
        {
            "n_train": len(y_train),
            "n_test": len(y_test),
            "test_size": test_size,
            "random_state": random_state,
            "cv_folds": cv,
            "test_auc_mean": y_test.mean(),
            "test_auc_std": y_test.std(),
            "baseline_rmse": baseline_rmse,
        })

    test_set_summary.to_csv(drug_folder / "test_set_summary.csv",header=["value"],)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        f"\n{'=' * 65}"
        f"\n{drug_name} - MODEL SCREEN"
        f"\n{'=' * 65}")

    display_columns = [
        "MAE",
        "RMSE",
        "R2",
        "CV_RMSE",
        "Train_R2",
        "RMSE_over_test_SD",
        "Improvement_vs_mean_%",
    ]

    print(results_df[display_columns].round(4).to_string())

    print("\nBest parameters:")

    for model_name in specs:
        print(f"{model_name}: " f"{results_df.loc[model_name, 'best_params']}")

    print(f"\nResults saved to:\n" f"{drug_folder.resolve()}")

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "drug": drug_name,
        "qc": qc,
        "distribution": distribution,
        "model_results": results_df,
        "models": fitted_models,
        "predictions": predictions,
        "modeling_data": data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results_folder": drug_folder,
    }