# Real Data Experiments

This directory contains scripts for conducting macroeconomic experiments on FRED-MD time series data.

### File Descriptions

| File | Description |
| :--- | :--- |
| `data_process.R` | Loads raw `.csv` data and performs time series processing and transformations. |
| `real_data_band.R` | Executes the BAND framework experiments and saves results to `.rds`. |
| `real_data_qr.R` | Executes Linear Quantile Regression experiments and saves results to `.rds`. |
| `real_data_qrf.R` | Executes Quantile Regression Forest experiments and saves results to `.rds`. |
| `comparison_statistics.R` | Loads results from all methods to generate the comparison tables. |
| `make_raw_plot.R` | Generates diagnostic and results plots for the paper appendix. |

**Note:** Certain specific BAND-related statistics mentioned in the paper were calculated manually from the generated model objects. All scripts are prepared for anonymous submission.