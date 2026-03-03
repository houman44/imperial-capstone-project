import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def load_data(X_file='X_data.txt', y_file='y_data.txt', n_dims=None):
    """
    Load X and y data from files.
    
    Args:
        X_file: Path to input features file
        y_file: Path to objective values file
        n_dims: Number of dimensions to use (None = use all available)
    
    Returns:
        X, y as numpy arrays
    """
    try:
        # Try loading X data
        X = np.loadtxt(X_file)
        
        # If X is 1D, reshape it
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Determine dimensions to use
        available_dims = X.shape[1]
        if n_dims is None:
            n_dims = available_dims
            print(f"Auto-detected {n_dims} dimensions from data")
        elif n_dims > available_dims:
            print(f"Warning: Requested {n_dims} dimensions but data has only {available_dims}")
            print(f"Using all {available_dims} available dimensions")
            n_dims = available_dims
        elif n_dims < available_dims:
            print(f"Using first {n_dims} dimensions from {available_dims} total dimensions")
            X = X[:, :n_dims]
        
        # Load y data
        y = np.loadtxt(y_file)
        
        # Validate dimensions match
        if len(y) != X.shape[0]:
            raise ValueError(f"Mismatch: X has {X.shape[0]} samples, y has {len(y)} samples")
        
        print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
        print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
        
        return X, y
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        return None, None


# Load data from files (or use example data)
X, y = load_data(args.x_file, args.y_file, n_dims=args.n_dims)

# Get number of dimensions from loaded data
n_dims = X.shape[1]

# Define kernel with optimization
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Fit Gaussian Process
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    alpha=1e-6,
    normalize_y=True
)

print("\nFitting Gaussian Process...")
gp.fit(X, y)

print(f"\nOptimized kernel: {gp.kernel_}")
print(f"Log marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")

# Expected Improvement acquisition function
def expected_improvement(X_sample, gp, y_best, xi=0.01):
    """
    Compute the Expected Improvement at X_sample.
    
    Args:
        X_sample: Point(s) at which to evaluate EI
        gp: Fitted GaussianProcessRegressor
        y_best: Best observed value
        xi: Exploration-exploitation trade-off parameter
    """
    X_sample = np.atleast_2d(X_sample)
    mu, sigma = gp.predict(X_sample, return_std=True)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    
    return ei

# Optimization objective (negative EI for minimization)
def neg_expected_improvement(X_sample, gp, y_best, xi=0.01):
    return -expected_improvement(X_sample, gp, y_best, xi)

# Find next sampling point using scipy.optimize
y_best = np.max(y)
print(f"\nBest observed value: {y_best:.6f}")

# Multi-start optimization
n_restarts = args.n_restarts
best_ei = -np.inf
best_x = None

print(f"\nOptimizing acquisition function with {n_restarts} random restarts...")

for i in range(n_restarts):
    # Random starting point
    x0 = np.random.uniform(0, 1, n_dims)
    
    # Optimize
    result = minimize(
        neg_expected_improvement,
        x0,
        args=(gp, y_best, args.xi),
        bounds=[(0, 1)] * n_dims,
        method='L-BFGS-B'
    )
    
    if -result.fun > best_ei:
        best_ei = -result.fun
        best_x = result.x

print(f"\n{'='*60}")
print("RECOMMENDED NEXT SAMPLING POINT:")
print(f"{'='*60}")
print(f"Point: [{', '.join([f'{x:.6f}' for x in best_x])}]")
print(f"Expected Improvement: {best_ei:.6f}")

# Predict at the new point
mu_new, sigma_new = gp.predict(best_x.reshape(1, -1), return_std=True)
print(f"Predicted mean: {mu_new[0]:.6f}")
print(f"Predicted std: {sigma_new[0]:.6f}")
print(f"95% confidence interval: [{mu_new[0] - 1.96*sigma_new[0]:.6f}, {mu_new[0] + 1.96*sigma_new[0]:.6f}]")

# Visualization
print("\nGenerating visualizations...")

# Create a grid for visualization (2D slices) - only if n_dims >= 2
if n_dims >= 2:
    n_grid = 50
    x1_grid = np.linspace(0, 1, n_grid)
    x2_grid = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # Fix remaining dimensions at best_x values
    if n_dims == 2:
        X_grid = np.c_[X1.ravel(), X2.ravel()]
    else:
        # For higher dimensions, fix dims 3+ at best_x values
        fixed_dims = np.tile(best_x[2:], (n_grid**2, 1))
        X_grid = np.c_[X1.ravel(), X2.ravel(), fixed_dims]

    # Predict mean and std
    mu_grid, sigma_grid = gp.predict(X_grid, return_std=True)
    mu_grid = mu_grid.reshape(n_grid, n_grid)
    sigma_grid = sigma_grid.reshape(n_grid, n_grid)

    # Calculate EI on grid
    ei_grid = expected_improvement(X_grid, gp, y_best, args.xi).reshape(n_grid, n_grid)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 5))

    # Plot 1: GP Mean prediction
    ax1 = fig.add_subplot(131)
    im1 = ax1.contourf(X1, X2, mu_grid, levels=20, cmap='viridis')
    ax1.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='black', cmap='viridis', 
                vmin=mu_grid.min(), vmax=mu_grid.max(), linewidths=2, zorder=5)
    ax1.scatter(best_x[0], best_x[1], c='red', s=300, marker='*', 
                edgecolors='black', linewidths=2, zorder=6, label='Next point')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    if n_dims > 2:
        fixed_str = ', '.join([f'X{i+1}={best_x[i]:.3f}' for i in range(2, n_dims)])
        ax1.set_title(f'GP Mean Prediction ({fixed_str})')
    else:
        ax1.set_title('GP Mean Prediction')
    plt.colorbar(im1, ax=ax1)
    ax1.legend()

    # Plot 2: GP Uncertainty (std)
    ax2 = fig.add_subplot(132)
    im2 = ax2.contourf(X1, X2, sigma_grid, levels=20, cmap='plasma')
    ax2.scatter(X[:, 0], X[:, 1], c='white', s=100, edgecolors='black', 
                linewidths=2, zorder=5)
    ax2.scatter(best_x[0], best_x[1], c='red', s=300, marker='*', 
                edgecolors='black', linewidths=2, zorder=6, label='Next point')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    if n_dims > 2:
        ax2.set_title(f'GP Uncertainty (Std) ({fixed_str})')
    else:
        ax2.set_title('GP Uncertainty (Std)')
    plt.colorbar(im2, ax=ax2)
    ax2.legend()

    # Plot 3: Expected Improvement
    ax3 = fig.add_subplot(133)
    im3 = ax3.contourf(X1, X2, ei_grid, levels=20, cmap='coolwarm')
    ax3.scatter(X[:, 0], X[:, 1], c='black', s=100, edgecolors='white', 
                linewidths=2, zorder=5)
    ax3.scatter(best_x[0], best_x[1], c='red', s=300, marker='*', 
                edgecolors='black', linewidths=2, zorder=6, label='Next point')
    ax3.set_xlabel('X1')
    ax3.set_ylabel('X2')
    if n_dims > 2:
        ax3.set_title(f'Expected Improvement ({fixed_str})')
    else:
        ax3.set_title('Expected Improvement')
    plt.colorbar(im3, ax=ax3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('bayesian_optimization_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'bayesian_optimization_results.png'")
else:
    print("\n1D data - skipping 2D visualizations")

# 3D scatter plot of observations (only if 3D data)
if n_dims >= 3:
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=100, 
                         cmap='viridis', edgecolors='black', linewidths=1)
    ax.scatter(best_x[0], best_x[1], best_x[2], c='red', s=400, marker='*', 
               edgecolors='black', linewidths=2, label='Next point')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Observations in 3D Space')
    plt.colorbar(scatter, ax=ax, label='Objective Value')
    ax.legend()
    plt.savefig('observations_3d.png', dpi=300, bbox_inches='tight')
    print("3D visualization saved as 'observations_3d.png'")

plt.show()

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")
print(f"- OBJECTIVE: MAXIMIZATION")
print(f"- Fitted GP with optimized hyperparameters")
print(f"- Used {n_restarts} random restarts for acquisition optimization")
print(f"- Best EI found: {best_ei:.6f}")
print(f"- This point balances exploration (high uncertainty) and exploitation (high predicted value)")
print(f"- Sample this point next to find HIGHER values!")
print(f"\nData files:")
print(f"  {args.x_file} - Input features (create this file with your data)")
print(f"  {args.y_file} - Objective values (create this file with your data)")
print(f"\nUsage examples:")
print(f"  python script.py --n_dims 5 --x_file my_X.txt --y_file my_y.txt")
print(f"  python script.py --n_dims 3 --n_restarts 50 --xi 0.05")