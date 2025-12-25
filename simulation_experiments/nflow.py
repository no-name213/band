import numpy as np
import torch
from torch import optim
import pandas as pd
import itertools
import time
from sklearn.model_selection import train_test_split


# conda create -n nflow_env python=3.10
# conda activate nflow_env
# Install dependencies:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
# pip install nflows
# pip install numpy pandas matplotlib seaborn scipy scikit-learn


# --- Attempting to import nflows components. Assumes nflows is installed. ---
try:
    from nflows import transforms, distributions, flows
except ImportError:
    print("Warning: nflows library not found. Using Mock objects for demonstration.")
    # Fallback to Mock objects if nflows is not installed (prevents script crash)
    class MockStandardNormal(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.loc = torch.zeros(shape)
            self.scale = torch.ones(shape)
        def log_prob(self, x): return torch.zeros(x.shape[0])
        def sample(self, num_samples): return torch.randn(num_samples, *self.loc.shape)

    class MockFlow(torch.nn.Module):
        def __init__(self, transform, distribution):
            super().__init__()
            self.transform = transform
            self.distribution = distribution
        def log_prob(self, x): return torch.zeros(x.shape[0])
        def sample(self, num_samples): 
            z = self.distribution.sample(num_samples)
            return z * 2
        def train(self): pass
        def eval(self): pass

    class MockTransforms:
        class MaskedAffineAutoregressiveTransform(torch.nn.Module):
            def __init__(self, features, hidden_features): super().__init__()
        class RandomPermutation(torch.nn.Module):
            def __init__(self, features): super().__init__()
        class CompositeTransform(torch.nn.Module):
            def __init__(self, layers): super().__init__()
            
    transforms = MockTransforms()
    distributions = MockStandardNormal
    flows = MockFlow
# --------------------------------------------------------------------------


# ==============================================================================
# 1. UTILITY FUNCTIONS (ECDF, CW DISTANCE, SAMPLING)
# ==============================================================================

def ecdf_grid(values, z_grid):
    """
    Compute the empirical CDF of `values` at points `z_grid`.
    """
    values = np.sort(values)
    n = len(values)
    # np.searchsorted is highly efficient for ECDF computation
    cdf = np.searchsorted(values, z_grid, side='right') / n
    return cdf

# X = samples 
# Y = x_test
def cramer_wold_distance(X,
                         Y,
                         N_sparse=10000,
                         N_dense=10000,
                         z_grid = np.arange(-20, 20, 0.02),
                         seed=None):
    """
    Compute the Cramer-Wold distance using a mix of sparse (2-coordinate emphasis)
    and dense random projections, plus coordinate axes projections.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n_x, p = X.shape
    n_y, _ = Y.shape
    

    rng = np.random.default_rng(seed)

    # --- A. Sparse Projections (2-Coordinate Emphasis) ---
    proj_sparse = np.zeros((N_sparse, p))
    for i in range(N_sparse):
        indices = rng.choice(p, 2, replace=False)
        proj_sparse[i, indices] = rng.normal(size=2)
    norms_sparse = np.linalg.norm(proj_sparse, axis=1, keepdims=True)
    norms_sparse[norms_sparse == 0] = 1 
    proj_sparse /= norms_sparse
    
    # --- B. Dense Projections (Full Random Directions) ---
    proj_dense = rng.normal(size=(N_dense, p))
    proj_dense /= np.linalg.norm(proj_dense, axis=1, keepdims=True)

    # --- C. Projections onto Coordinate Axes (Basis Vectors) ---
    proj_axes = np.eye(p)
    
    # 3) Combine all projection matrices
    proj_combined = np.concatenate((proj_sparse, proj_dense), axis=0) 
    X_proj_rand = X @ proj_combined.T
    Y_proj_rand = Y @ proj_combined.T

    # 4) Add Coordinate-wise projections
    X_proj_axes = X @ proj_axes.T 
    Y_proj_axes = Y @ proj_axes.T 

    X_proj_all = np.concatenate((X_proj_rand, X_proj_axes), axis=1)
    Y_proj_all = np.concatenate((Y_proj_rand, Y_proj_axes), axis=1)

    K = X_proj_all.shape[1]
    
    # 5) Compute CDF differences (KS Distance) for all K projections
    max_diff = 0.0
    for k in range(K):
        cdfs_X = ecdf_grid(X_proj_all[:, k], z_grid)
        cdfs_Y = ecdf_grid(Y_proj_all[:, k], z_grid)
        current_diff = np.max(np.abs(cdfs_X - cdfs_Y))
        if current_diff > max_diff:
            max_diff = current_diff

    # 6) Final CW distance = Maximum Discrepancy
    return max_diff

@torch.no_grad()
def evaluate_nll(flow, data):
    """
    Computes the Negative Log-Likelihood (NLL) for a given dataset.
    Used for hyperparameter tuning on the validation set.
    """
    flow.eval()
    return -flow.log_prob(data).mean().item()


# ==============================================================================
# 2. DATA GENERATION FUNCTIONS
# ==============================================================================

def sample_gmm(n=1000, p=2, d=2.0, var=1, seed=None):
    """Generate data in R^p from a Bimodal Gaussian Mixture."""
    if seed is not None:
        torch.manual_seed(seed)
    
    cov = torch.eye(p) * var
    mean1 = torch.zeros(p)
    mean2 = torch.ones(p) * d
    means = [mean1, mean2]

    comps = torch.randint(0, 2, (n,))
    x = torch.stack([
        torch.distributions.MultivariateNormal(means[i], cov).sample()
        for i in comps
    ])
    return x

def sample_bimodal_uniform(n=1000, p=2, d=2.0, width=1.5, seed=None):
    """Generate data in R^p from a Bimodal Uniform distribution."""
    if seed is not None:
        torch.manual_seed(seed)

    n1 = n // 2
    n2 = n - n1
    x1 = (torch.rand(n1, p) * 2 * width - width)
    x2 = (torch.rand(n2, p) * 2 * width - width) + d
    return torch.cat([x1, x2], dim=0)

# ==============================================================================
# 3. NORMALIZING FLOW FUNCTION (with suppressed printing for clean loop)
# ==============================================================================

def train_normalizing_flow(inputs,
                           num_layers,
                           hidden_features=64,
                           batch_size=32,
                           num_epochs=5000,
                           lr=1e-3,
                           verbose=False): # Added verbose flag
    """
    Defines, trains, and returns a Masked Autoregressive Flow (MAF) model.
    """
    if inputs is None or inputs.numel() == 0:
        raise ValueError("Inputs tensor cannot be None or empty.")

    p = inputs.shape[1]

    # 1. Define the transformation layers (MAF + Permutation)
    layers = []
    for _ in range(num_layers):
        layers.append(
            transforms.MaskedAffineAutoregressiveTransform(
                features=p,
                hidden_features=hidden_features
            )
        )
        layers.append(transforms.RandomPermutation(features=p))

    transform = transforms.CompositeTransform(layers)
    base_distribution = distributions.StandardNormal(shape=[p])
    flow = flows.Flow(transform=transform, distribution=base_distribution)
    
    # 2. Fit / Train the Flow
    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    flow.train()

    if verbose:
        print(f"Starting training for {num_epochs} epochs with {inputs.shape[0]} samples...")
        
    for epoch in range(num_epochs):
        # Select random batch indices
        idx = torch.randperm(inputs.size(0))[:batch_size]
        batch = inputs[idx]
        
        # Calculate loss (Negative Log-Likelihood)
        loss = -flow.log_prob(batch).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % (num_epochs // 10) == 0:
            print(f'Epoch {epoch+1}, Negative Log-Likelihood: {loss.item():.4f}')

    flow.eval()
    return flow

# ==============================================================================
# 4. SIMULATION STUDY EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # --- Define Simulation Parameters ---
    P_VALUES = [2, 10, 30]
    N_VALUES = [100, 1000]
    D_VALUES = [0.0, 2, 4] # Use floats for d parameter
    MODEL_TYPES = ['gaussian', 'unit'] # Use 'gaussian' and 'unit' as model strings
    N_LAYERS_VALUES = [2, 3, 4, 6] # Varying number of flow layers (model depth)
    
    # R_REPETITIONS is the number of independent samples generated for each parameter set.
    R_REPETITIONS = 30

    
    # Flow Hyperparameters (Fixed)
    HIDDEN_FEATURES = 64 
    BATCH_SIZE = 32      
    NUM_EPOCHS = 5000    
    
    # Test/Evaluation Parameters
    VALIDATION_SPLIT_RATIO = 0.3 # 30% for validation in Phase 1
    NUM_SAMPLES_FLOW = 10000
    N_TEST = 10000
    
    # Results storage
    results_list = []
    validation_list = []
    
    # Parameter space for Unique Experiments
    unique_experiments = list(itertools.product(P_VALUES, N_VALUES, D_VALUES, MODEL_TYPES))
    
    # Dictionary to store the fixed samples S_N for each unique repetition/experiment
    fixed_samples = {}
    
    # --- PHASE 0: DATA GENERATION (Generate and fix R_REPETITIONS independent samples) ---
    print("--- PHASE 0: Generating and Fixing R_REPETITIONS independent samples per Parameter Set ---")
    
    for p, n, d, model in unique_experiments:
        # Loop over the number of independent samples we want to generate
        for r_ in range(1, R_REPETITIONS + 1):
            exp_key = (p, n, d, model, r_) # New 5-tuple key: (p, n, d, model, repetition_id)
            
            if model == 'unit':
                inputs_full = sample_bimodal_uniform(n=n, p=p, d=d)
            else: # 'gaussian'
                inputs_full = sample_gmm(n=n, p=p, d=d)
            
            fixed_samples[exp_key] = inputs_full
    
    # --- PHASE 1: TUNING AND HYPERPARAMETER SELECTION (EXPLORATION) ---
    print("\n--- PHASE 1: Starting Tuning/Exploration (Finding Best N_LAYERS via 70:30 split of R_REPETITIONS independent samples) ---")
    
    # Get all keys (all unique samples) to iterate through
    all_sample_keys = list(fixed_samples.keys())
    total_experiments = len(all_sample_keys)
    current_experiment = 0

    # Outer Loop: Iterate over all unique sample realizations (p, n, d, model, r_)
    for p, n, d, model, r_ in all_sample_keys:
        current_experiment += 1
        exp_key = (p, n, d, model, r_)
        inputs_full = fixed_samples[exp_key]
        
        print(f"\n--- Sample Run {current_experiment}/{total_experiments} (r={r_}): P={p}, N={n}, D={d}, Model={model} ---")
        
        # --- 1. Perform a SINGLE 70:30 split on this unique sample S_N^(r) ---
        # A fixed random_state is used since the variance is handled by the R_REPETITIONS unique samples.
        inputs_train_np, inputs_val_np = train_test_split(
            inputs_full.cpu().numpy(), 
            test_size=VALIDATION_SPLIT_RATIO, 
            random_state=42 
        )
        inputs_train = torch.from_numpy(inputs_train_np).float()
        inputs_val = torch.from_numpy(inputs_val_np).float()

        # Inner Loop: Iterate over model depths (N_LAYERS_VALUES)
        for n_layers in N_LAYERS_VALUES:
            # print(f"  [L={n_layers}] Training...") # Suppressing print
            
            # --- 2. Train the nflow model on the 70% training set ---
            start_time = time.time()
            flow = train_normalizing_flow(
                inputs=inputs_train,
                num_layers=n_layers,
                hidden_features=HIDDEN_FEATURES,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                verbose=False
            )
            train_duration = time.time() - start_time
            
            # --- 3. Evaluate NLL on the 30% validation set ---
            val_nll = evaluate_nll(flow, inputs_val)
            
            # --- 4. Store Results (Tuning metric: NLL) ---
            result_entry = {
                'repetition': r_, # The ID of the independent sample realization
                'model': model,
                'tuning_metric': val_nll,
                'p': p,
                'n': n,
                'd': d,
                'n_layers': n_layers,
                'train_time_sec': train_duration
            }
            results_list.append(result_entry)
    
    # --- DETERMINE OPTIMAL N_LAYERS PER SAMPLE REALIZATION ---
    results_df = pd.DataFrame(results_list)
    
    # Group by the full 5-tuple key and find the best layer for EACH repetition
    idx = results_df.groupby(['model', 'p', 'n', 'd', 'repetition'])['tuning_metric'].idxmin()
    best_layers_df = results_df.loc[idx].reset_index(drop=True)
    best_layers_df = best_layers_df.rename(columns={'tuning_metric': 'best_tuning_nll'})
    
    best_layers_df = best_layers_df[['model', 'p', 'n', 'd', 'repetition', 'n_layers', 'best_tuning_nll']]
    best_layers_df = best_layers_df.rename(columns={'n_layers': 'best_n_layers'})
    
    print("\n--- PHASE 1 COMPLETE: Best N_LAYERS selected for EACH of the R_REPETITIONS independent samples. ---")


    # --- PHASE 2: INDEPENDENT VALIDATION (Unbiased CW Distance) ---
    print("\n--- PHASE 2: Starting Independent Validation (Testing best N_LAYERS with CW Distance) ---")

    
    validation_combinations = best_layers_df.to_dict('records')
    total_experiments = len(all_sample_keys)
    
    
    # Outer Loop: Iterate over all unique (p, n, d, model, r_) combinations (the selected best model)
    for combo in validation_combinations:
        
        p = combo['p']
        n = combo['n']
        d = combo['d']
        model = combo['model']
        r_ = combo['repetition']
        n_layers = combo['best_n_layers']
        
        exp_key = (p, n, d, model, r_)

        # --- 1. Retrieve the FULL FIXED Sample (S_N^(r)) for training ---
        inputs = fixed_samples[exp_key].float()
        

        train_duration = 0.0
        


            
        start_time = time.time()
        flow = train_normalizing_flow(
            inputs=inputs, 
            num_layers=n_layers, 
            hidden_features=HIDDEN_FEATURES,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            verbose=False
        )
        train_duration = time.time() - start_time
        
        # --- 3. Sample from the trained density ---
        with torch.no_grad():
            samples = flow.sample(NUM_SAMPLES_FLOW).detach().cpu().numpy()
        
        # --- 4. Generate INDEPENDENT Test Sample (x_test) ---
        # Use r_validation + offset as seed to ensure test sample changes each time
        if model == 'unit':
            x_test = sample_bimodal_uniform(n=N_TEST, p=p, d=d).cpu().numpy()
        else: # 'gaussian'
            x_test = sample_gmm(n=N_TEST, p=p, d=d).cpu().numpy()
        
        # --- 5. Calculate Cramer-Wold Distance (Error) ---
        err = cramer_wold_distance(samples, x_test)

            


        # --- 6. Store Validation Results (Aggregated per Repetition) ---
        validation_entry = {
            'repetition': r_,
            'model': model,
            'p': p,
            'n': n,
            'd': d,
            'best_n_layers': n_layers,
            # Standard Metrics
            'err': err,
        }
        validation_list.append(validation_entry)

    # 1. Convert to DataFrame
    validation_df_final = pd.DataFrame(validation_list)
    
    # --- NEW STEP: SAVE THE RAW DATA FOR R PROCESSING ---
    # This DataFrame contains all results before averaging across repetitions.
    raw_output_filename = "simulation_results/nflow_monte_carlo.csv"
    try:
        validation_df_final.to_csv(raw_output_filename, index=False)
        print(f"Full, unaggregated raw data saved to {raw_output_filename}.")
    except Exception as e:
        print(f"Error saving raw data to CSV file: {e}")
        
        
    