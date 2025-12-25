# Simulation Experiments

This directory contains scripts for evaluating the **BAND** framework against synthetic benchmarks, focusing on accuracy, sparsity, and distributional fitting.

### File Descriptions

| File | Description |
| :--- | :--- |
| `XXX_monte_carlo.R` | Runs R-based simulation experiments for Cramér–Wold (CW) distance comparisons. |
| `nflow.py` | Runs Python-based Neural Flow experiments for CW distance comparisons. |
| `XXX_sparsity.R` | Runs R-based simulations for the blockwise Gaussian sparsity experiment. |
| `nflow_sparsity.py` | Runs Python-based Neural Flow simulations for the sparsity experiment. |
| `comparison_accuracy.R` | Processes results to generate CW distance comparison tables for both DGPs. |
| `comparison_sparsity.R` | Processes results to generate tables for the blockwise Gaussian sparsity experiment. |

**Notes:**

*   **Figure Generation:** To generate figures comparing ground truth data against the distributions fitted by the four methods, users may perform sampling using the fitted models provided in the experimental scripts.
*   **Python Dependency:** The `nflow` baseline is implemented in Python and is required to generate the corresponding comparison data.
*   **Anonymity:** All scripts and documentation have been prepared for **anonymous submission**.
*   **Execution:** All R scripts should be run from the project root using the provided `.Rproj` file to ensure correct path resolution.