from pathlib import Path
import pandas as pd
from drug_screening import screen_drug


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPRESSION_PATH = (DATA_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
DRUG_RESPONSE_PATH = (DATA_DIR / "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv")
DRUG_COUNTS_PATH = (PROCESSED_DIR / "drug_counts_overlap.csv")
SUMMARY_OUTPUT_PATH = (RESULTS_DIR / "multi_drug_screening_summary.csv")


# ============================================================
# MULTI-DRUG SCREENING
# ============================================================
#
# Purpose:
# Screen multiple PRISM drug-response phenotypes to determine
# whether their AUC values can be predicted from baseline
# RNA-seq gene expression.
#
# The same screen_drug() pipeline is applied independently to
# every drug. This standardizes:
#
#   - PRISM measurement selection
#   - RNA-seq profile selection
#   - train/test splitting
#   - cross-validation
#   - model hyperparameter tuning
#   - baseline comparison
#   - evaluation metrics
#
# This stage is exploratory. Its purpose is to compare the
# predictability of different drugs under a common modeling
# procedure rather than to produce final optimized models.
#
# Drugs with stronger and more consistent predictive performance
# can subsequently be selected for deeper biological and
# methodological investigation.


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("MULTI-DRUG SCREENING")
print("=" * 60)

expression = pd.read_csv(EXPRESSION_PATH)

drug_response = pd.read_csv(DRUG_RESPONSE_PATH)

drug_counts_overlap = pd.read_csv(DRUG_COUNTS_PATH)

print("RNA-seq dataset shape:", expression.shape)

print("PRISM dataset shape:", drug_response.shape)

print("Drug-overlap table shape:", drug_counts_overlap.shape)


# ============================================================
# 2. DEFINE DRUGS TO SCREEN
# ============================================================

# Drugs were selected primarily because they have relatively
# large numbers of cell lines represented in both the PRISM
# response dataset and the RNA-seq expression dataset.
#
# Maximizing sample overlap is particularly important because
# the modeling problem is highly dimensional:
#
#   ~19,000 gene-expression predictors
#   versus several hundred cell lines per drug.

DRUGS_TO_SCREEN = [
    "AZD2014",
    "cinacalcet",
    "osimertinib",
    "teniposide",
    "floxuridine",
    "tanespimycin",
    "ponatinib",
    "BMS-754807",
    "pevonedistat",
    "talazoparib",
    "trametinib",
    "alvespimycin",
    "everolimus",
]

print("\nNumber of drugs selected:", len(DRUGS_TO_SCREEN))
print("\nDrugs selected for screening:")

for drug in DRUGS_TO_SCREEN:
    print(f"  - {drug}")


# ============================================================
# 3. RUN STANDARDIZED SCREENING
# ============================================================

# Elastic Net and RBF-SVR are enabled for the initial screen.
#
# Random Forest is disabled here because it is substantially
# more computationally expensive in the high-dimensional
# expression dataset.
#
# It can be enabled later for selected drugs by setting:
#
#     rf=True

all_results = []

failed_drugs = []

for i, drug_name in enumerate(DRUGS_TO_SCREEN,start=1):

    print("\n" + "=" * 60)
    print(f"SCREENING {i}/{len(DRUGS_TO_SCREEN)}: "f"{drug_name}")

    print("=" * 60)
    try:

        drug_results = screen_drug(
            expression=expression,
            drug_response=drug_response,
            drug_name=drug_name,
            enet=True,
            svr=True,
            rf=False,
            show_plots=False,
        )

        # screen_drug() should return a DataFrame containing
        # one row for each model evaluated for this drug.
        if drug_results is not None:
            drug_results = drug_results.copy()

            # Add the drug name if the reusable function does
            # not already include it.
            if "Drug" not in drug_results.columns:
                drug_results.insert(
                    0,
                    "Drug",
                    drug_name
                )

            all_results.append(
                drug_results
            )

    except Exception as error:

        print(
            f"\nScreening failed for {drug_name}: "
            f"{error}"
        )

        failed_drugs.append(
            {
                "Drug": drug_name,
                "Error": str(error),
            }
        )


# ============================================================
# 4. COMBINE SCREENING RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MULTI-DRUG SCREENING SUMMARY")
print("=" * 60)

if all_results:

    screening_summary = pd.concat(
        all_results,
        ignore_index=True
    )

    screening_summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False
    )

    print(
        screening_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved combined screening results to:",
        SUMMARY_OUTPUT_PATH
    )

else:

    screening_summary = pd.DataFrame()

    print(
        "No screening results were returned."
    )


# ============================================================
# 5. REPORT FAILED DRUGS
# ============================================================

if failed_drugs:

    failed_drugs_df = pd.DataFrame(failed_drugs)

    failed_output_path = (RESULTS_DIR / "multi_drug_screening_failures.csv")

    failed_drugs_df.to_csv(failed_output_path, index=False)

    print("\nDrugs that could not be screened:")
    print(failed_drugs_df.to_string(index=False))
    print("\nFailure report saved to:", failed_output_path)

else:
    print("\nAll selected drugs completed successfully.")

# ============================================================
# 6. SCREENING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("SCREENING COMPLETE")
print("=" * 60)
print(f"Requested drugs: {len(DRUGS_TO_SCREEN)}")
print(f"Successful drugs: " f"{len(DRUGS_TO_SCREEN) - len(failed_drugs)}")
print(f"Failed drugs: {len(failed_drugs)}")