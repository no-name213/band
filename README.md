# BAND: Bayesian Network Distribution Regression

**BAND** is a nonparametric framework for modeling complex multivariate distributions. It leverages gradient-boosted trees (via **CatBoost**) to fit a sequence of conditional distributions through an autoregressive structure, enabling the construction of high-density confidence regions without assuming Gaussianity or specific functional forms.

## 🚀 Installation

### R Dependencies
1. **Open the Project**: Open `band.Rproj` in RStudio to ensure the `here` package identifies the project root.
2. **Install R Packages**:
   ```r
   # Core dependencies and Baseline benchmarks
   install.packages(c("here", "quantreg", "quantregForest", "mclust", 
                      "MASS", "kdevine", "VineCopula", "remotes"))
   
   # Install CatBoost (requires remotes)
   remotes::install_github('catboost/catboost', subdir = 'catboost/R-package')
   ```
   
### Python Dependencies (for Normalizing Flow Baselines)
For simulation experiments involving Neural Flows, install the following via `conda` or `pip`:
```bash
pip install torch nflows numpy pandas scikit-learn
```

## 💻 Usage Example

The following example demonstrates how to train a BAND model on synthetic Gaussian data, construct a conditional confidence region, and label test samples.

```r
library(here)
source(here("train.R"))
source(here("confidence_region.R"))

# 1. Setup Data (Standard Gaussian)
n <- 200; p <- 3
x_train <- matrix(rnorm(n * p), nrow = n, ncol = p)

# 2. Train Model
# Returns a sampler function with model metadata attached as attributes
band_sampler <- band(x_train, factor = 1.0)

# 3. Unconditional Sampling
# Generate 100 synthetic observations
x_synth <- band_sampler(n_sampling = 100)

# 4. Conditional Confidence Region
# Define conditioning (x2 = 0.7); use NA for unobserved coordinates
x_condi <- c(NA, 0.7, NA)
region  <- confidence_region(band_sampler, x_condi = x_condi, alpha_level = 0.9)

# 5. Membership Labeling
# Generate test data and check if points fall within the 90% region
x_test <- matrix(rnorm(100 * p), nrow = 100, ncol = p)

# Labels: 1 = Inside Region, 0 = Outside
labels <- cluster_labels(
  x = x_test, 
  Z = region, 
  band_sampler = band_sampler, 
  x_condi_coordinates = 1 - is.na(x_condi) # Binary mask of conditioned indices
)

# Summary of results
table(labels)
```

## 📂 Project Structure

### Core Methodology
*   `train.R`: Main interface for training the BAND framework and hyperparameter management.
*   `sampling.R`: Logic for generating synthetic one-hot vectors and multivariate samples.
*   `confidence_region.R`: Inference tools for constructing high-density regions and calculating cluster membership.
*   `util.R`: Helpers for interval binning, one-hot encoding/decoding, and distance metrics.
*   `dgp.R`: Data Generating Processes for synthetic benchmarking.

### Experiments & Applications
*   `simulation_experiments/`: Scripts evaluating BAND performance against synthetic ground truths.
*   `real_data/`: Macroeconomic experiments using the FRED-MD database, featuring comparative analysis with Quantile Regression (QR) and Quantile Regression Forests (QRF).

## 🛠 Key Features
*   **Fully Nonparametric**: Effectively captures multi-modal, skewed, and non-linear dependencies.
*   **Flexible Inference**: Supports joint, conditional, and marginalized probability queries within a unified framework.
*   **High-Density Regions**: Identifies the smallest set of hypercubes covering a target probability mass.

## 📄 Note on Anonymity
This repository is prepared for **anonymous submission**. All author names and institutional affiliations have been removed from the source code and documentation.