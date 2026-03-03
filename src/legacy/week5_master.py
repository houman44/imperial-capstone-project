"""
Week 5 Hybrid Bayesian Optimization System
===========================================

Complete implementation using:
- Pure GP (F1, F2, F3)
- GP + SVM (F4, F6, F7)
- GP + Gradients (F5)
- Full Hybrid (F8)

Run this to get next query recommendations for all functions.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WEEK 5 HYBRID BAYESIAN OPTIMIZATION")
print("="*80)

# ============================================================================
# ALL DATA (Weeks 1-5)
# ============================================================================

# Function 1 (2D) - MAXIMIZE
X1 = np.array([
    [0.19, 0.04], [0.76, 0.54], [0.44, 0.80], [0.35, 0.32],
    [0.30, 0.32], [0.57, 0.03], [0.57, 0.18], [0.83, 0.79],
    [0.61, 0.84], [0.04, 0.29], [0.88, 0.29], [0.96, 0.03],
    [0.58, 0.54], [0.580000, 0.540000]  # Week 5
])
y1 = np.array([
    0.26, 0.08, 5.83, 7.76, 5.18, 0.0, 0.0, 4.03,
    5.84, 0.0, 0.0, 0.0, 0.02, 5.494870748162966e-8
])

# Function 2 (2D) - MAXIMIZE  
X2 = np.array([
    [0.19, 0.04], [0.76, 0.54], [0.44, 0.80], [0.35, 0.32],
    [0.30, 0.32], [0.57, 0.03], [0.57, 0.18], [0.83, 0.79],
    [0.61, 0.84], [0.04, 0.29], [0.88, 0.29], [0.96, 0.03],
    [0.70, 0.93], [0.700000, 0.930000]  # Week 4, Week 5
])
y2 = np.array([
    0.40, -0.09, 0.23, 0.68, 0.43, -0.23, 0.12, 0.18,
    0.22, -0.06, 0.09, -0.09, 0.49, 0.49954705055291193
])

# Function 3 (3D) - MINIMIZE
X3 = np.array([
    [0.19, 0.04, 0.61], [0.76, 0.54, 0.66], [0.44, 0.80, 0.21],
    [0.35, 0.32, 0.12], [0.30, 0.32, 0.62], [0.57, 0.03, 0.88],
    [0.57, 0.18, 0.34], [0.83, 0.79, 0.11], [0.61, 0.84, 0.98],
    [0.04, 0.29, 0.98], [0.88, 0.29, 0.41], [0.96, 0.03, 0.09],
    [0.93, 0.85, 0.64], [0.85, 0.94, 0.48], [0.99, 0.90, 0.91],
    [0.70, 0.75, 0.35], [0.700000, 0.300000, 0.600000]  # Week 5
])
y3 = np.array([
    -0.27, -0.50, -0.77, -0.41, -0.30, -0.70, -0.40, -0.69,
    -0.74, -0.73, -0.47, -0.63, -0.85, -0.83, -0.93,
    -0.79, -0.06382715038922124
])

# Function 4 (4D) - MINIMIZE
X4 = np.array([
    [0.19, 0.04, 0.61, 0.41], [0.76, 0.54, 0.66, 0.36],
    [0.44, 0.80, 0.21, 0.15], [0.35, 0.32, 0.12, 0.47],
    [0.30, 0.32, 0.62, 0.10], [0.57, 0.03, 0.88, 0.82],
    [0.57, 0.18, 0.34, 0.79], [0.83, 0.79, 0.11, 0.69],
    [0.61, 0.84, 0.98, 0.21], [0.04, 0.29, 0.98, 0.56],
    [0.88, 0.29, 0.41, 0.03], [0.96, 0.03, 0.09, 0.95],
    [0.93, 0.85, 0.64, 0.59], [0.85, 0.94, 0.48, 0.76],
    [0.99, 0.90, 0.91, 0.33], [0.30, 0.29, 0.46, 0.93],
    [0.90, 0.55, 0.95, 0.48], [0.50, 0.66, 0.31, 0.26],
    [0.06, 0.58, 0.76, 0.66], [0.67, 0.71, 0.77, 0.86],
    [0.22, 0.85, 0.88, 0.88], [0.26, 0.84, 0.95, 0.89],
    [0.00, 1.00, 1.00, 1.00], [0.00, 1.00, 1.00, 0.95],
    [0.55, 0.45, 0.52, 0.28], [0.54, 0.42, 0.55, 0.29],
    [0.51, 0.42, 0.46, 0.32], [0.51, 0.44, 0.47, 0.30],
    [0.51, 0.44, 0.46, 0.30], [0.51, 0.43, 0.46, 0.31],
    [0.530000, 0.470000, 0.380000, 0.300000]  # Week 5
])
y4 = np.array([
    -3.16, -3.00, -3.15, -3.03, -3.10, -2.95, -3.02, -2.98,
    -3.14, -3.07, -2.99, -2.92, -3.17, -3.19, -3.24, -3.04,
    -3.25, -3.08, -3.12, -3.22, -3.28, -3.29, -3.36, -3.36,
    -3.38, -3.38, -3.38, -3.38, -3.38, -3.38, -3.3897164075807775
])

# Function 5 (4D) - MAXIMIZE
X5 = np.array([
    [0.19, 0.04, 0.61, 0.41], [0.76, 0.54, 0.66, 0.36],
    [0.44, 0.80, 0.21, 0.15], [0.35, 0.32, 0.12, 0.47],
    [0.30, 0.32, 0.62, 0.10], [0.57, 0.03, 0.88, 0.82],
    [0.57, 0.18, 0.34, 0.79], [0.83, 0.79, 0.11, 0.69],
    [0.61, 0.84, 0.98, 0.21], [0.04, 0.29, 0.98, 0.56],
    [0.88, 0.29, 0.41, 0.03], [0.96, 0.03, 0.09, 0.95],
    [0.93, 0.85, 0.64, 0.59], [0.85, 0.94, 0.48, 0.76],
    [0.99, 0.90, 0.91, 0.33], [0.30, 0.29, 0.46, 0.93],
    [0.90, 0.55, 0.95, 0.48], [0.50, 0.66, 0.31, 0.26],
    [0.06, 0.58, 0.76, 0.66], [0.67, 0.71, 0.77, 0.86],
    [0.22, 0.85, 0.88, 0.88], [0.26, 0.84, 0.95, 0.89],
    [0.00, 1.00, 1.00, 1.00], [0.00, 1.00, 1.00, 0.95],
    [0.000000, 1.000000, 0.950000, 1.000000]  # Week 5
])
y5 = np.array([
    64.4, 18.3, 0.11, 109.6, 63.4, 258.4, 8.85, 4.21, 28.3, 55.5,
    0.51, 18.9, 113.7, 356.9, 67.1, 78.5, 432.9, 18.2, 44.3, 257.8,
    1088.9, 1550.9, 4440.5, 3819.7, 3819.7407576895994
])

# Function 6 (5D) - MINIMIZE
X6 = np.array([
    [0.19, 0.04, 0.61, 0.41, 0.93], [0.76, 0.54, 0.66, 0.36, 0.03],
    [0.44, 0.80, 0.21, 0.15, 0.12], [0.35, 0.32, 0.12, 0.47, 0.10],
    [0.30, 0.32, 0.62, 0.10, 0.50], [0.57, 0.03, 0.88, 0.82, 0.48],
    [0.57, 0.18, 0.34, 0.79, 0.66], [0.83, 0.79, 0.11, 0.69, 0.41],
    [0.61, 0.84, 0.98, 0.21, 0.21], [0.04, 0.29, 0.98, 0.56, 0.06],
    [0.88, 0.29, 0.41, 0.03, 0.88], [0.96, 0.03, 0.09, 0.95, 0.28],
    [0.93, 0.85, 0.64, 0.59, 0.36], [0.85, 0.94, 0.48, 0.76, 0.26],
    [0.99, 0.90, 0.91, 0.33, 0.03], [0.30, 0.29, 0.46, 0.93, 0.70],
    [0.90, 0.55, 0.95, 0.48, 0.07], [0.50, 0.66, 0.31, 0.26, 0.46],
    [0.06, 0.58, 0.76, 0.66, 0.30], [0.67, 0.71, 0.77, 0.86, 0.28],
    [0.22, 0.85, 0.88, 0.88, 0.13], [0.26, 0.84, 0.95, 0.89, 0.13],
    [0.00, 1.00, 1.00, 1.00, 0.10], [0.00, 1.00, 1.00, 0.95, 0.07],
    [0.55, 0.20, 0.75, 1.00, 0.03], [0.550000, 0.180000, 0.720000, 1.000000, 0.030000]  # Week 5
])
y6 = np.array([
    -0.58, -0.47, -0.49, -0.55, -0.50, -0.63, -0.57, -0.63,
    -0.56, -0.50, -0.50, -0.52, -0.67, -0.67, -0.58, -0.55,
    -0.63, -0.52, -0.58, -0.67, -0.64, -0.64, -0.64, -0.63,
    -0.64, -0.6368618203511591
])

# Function 7 (6D) - MAXIMIZE
X7 = np.array([
    [0.19, 0.04, 0.61, 0.41, 0.93, 0.29], [0.76, 0.54, 0.66, 0.36, 0.03, 0.12],
    [0.44, 0.80, 0.21, 0.15, 0.12, 0.10], [0.35, 0.32, 0.12, 0.47, 0.10, 0.50],
    [0.30, 0.32, 0.62, 0.10, 0.50, 0.48], [0.57, 0.03, 0.88, 0.82, 0.48, 0.66],
    [0.57, 0.18, 0.34, 0.79, 0.66, 0.41], [0.83, 0.79, 0.11, 0.69, 0.41, 0.21],
    [0.61, 0.84, 0.98, 0.21, 0.21, 0.06], [0.04, 0.29, 0.98, 0.56, 0.06, 0.88],
    [0.88, 0.29, 0.41, 0.03, 0.88, 0.28], [0.96, 0.03, 0.09, 0.95, 0.28, 0.36],
    [0.93, 0.85, 0.64, 0.59, 0.36, 0.26], [0.85, 0.94, 0.48, 0.76, 0.26, 0.03],
    [0.99, 0.90, 0.91, 0.33, 0.03, 0.70], [0.30, 0.29, 0.46, 0.93, 0.70, 0.07],
    [0.90, 0.55, 0.95, 0.48, 0.07, 0.46], [0.50, 0.66, 0.31, 0.26, 0.46, 0.30],
    [0.06, 0.58, 0.76, 0.66, 0.30, 0.28], [0.67, 0.71, 0.77, 0.86, 0.28, 0.13],
    [0.22, 0.85, 0.88, 0.88, 0.13, 0.13], [0.26, 0.84, 0.95, 0.89, 0.13, 0.10],
    [0.00, 1.00, 1.00, 1.00, 0.10, 0.07], [0.010000, 0.360000, 0.510000, 0.210000, 0.430000, 0.740000],
    [0.010000, 0.360000, 0.510000, 0.210000, 0.430000, 0.740000]  # Week 5
])
y7 = np.array([
    0.63, 0.17, 0.09, 0.22, 0.25, 0.55, 0.33, 0.29,
    0.45, 0.59, 0.13, 0.09, 0.56, 0.54, 0.70, 0.23,
    0.66, 0.24, 0.48, 0.69, 0.74, 0.78, 0.87, 1.89,
    1.8942067659620383
])

# Function 8 (8D) - MAXIMIZE
X8 = np.array([
    [0.19, 0.04, 0.61, 0.41, 0.93, 0.29, 0.74, 0.59],
    [0.76, 0.54, 0.66, 0.36, 0.03, 0.12, 0.10, 0.15],
    [0.44, 0.80, 0.21, 0.15, 0.12, 0.10, 0.50, 0.48],
    [0.030000, 0.070000, 0.015000, 0.050000, 1.000000, 0.850000, 0.550000, 0.920000]  # Week 5
])
y8 = np.array([64.4, 18.3, 0.11, 9.536385])

# Store all data
all_data = {
    'F1': (X1, y1, 'MAXIMIZE', 2),
    'F2': (X2, y2, 'MAXIMIZE', 2),
    'F3': (X3, y3, 'MINIMIZE', 3),
    'F4': (X4, y4, 'MINIMIZE', 4),
    'F5': (X5, y5, 'MAXIMIZE', 4),
    'F6': (X6, y6, 'MINIMIZE', 5),
    'F7': (X7, y7, 'MAXIMIZE', 6),
    'F8': (X8, y8, 'MAXIMIZE', 8),
}

print(f"\n📊 DATA LOADED:")
for func_name, (X, y, goal, dims) in all_data.items():
    best = y.max() if goal == 'MAXIMIZE' else y.min()
    print(f"   {func_name}: {len(X)} samples, {dims}D, {goal}, best={best:.4f}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def expected_improvement(X, gp, y_best, xi=0.01, maximize=True):
    """
    Compute Expected Improvement acquisition function
    """
    mu, sigma = gp.predict(X, return_std=True)
    mu = mu.reshape(-1)
    sigma = sigma.reshape(-1)
    
    if maximize:
        imp = mu - y_best - xi
    else:
        imp = y_best - mu - xi
    
    Z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    
    return ei

def compute_gradient(gp, x, epsilon=1e-5):
    """Compute gradient via finite differences"""
    x = np.array(x).reshape(1, -1)
    gradient = np.zeros(x.shape[1])
    
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[0, i] += epsilon
        x_minus[0, i] -= epsilon
        
        x_plus = np.clip(x_plus, 0, 1)
        x_minus = np.clip(x_minus, 0, 1)
        
        f_plus = gp.predict(x_plus)[0]
        f_minus = gp.predict(x_minus)[0]
        gradient[i] = (f_plus - f_minus) / (2 * epsilon)
    
    return gradient

def gradient_ascent(gp, x_start, learning_rate=0.05, max_steps=100, maximize=True):
    """Gradient ascent/descent on GP surrogate"""
    x = np.array(x_start).copy()
    
    for step in range(max_steps):
        grad = compute_gradient(gp, x)
        
        if maximize:
            x_new = x + learning_rate * grad
        else:
            x_new = x - learning_rate * grad
        
        x_new = np.clip(x_new, 0, 1)
        
        if np.linalg.norm(x_new - x) < 1e-4:
            break
        
        x = x_new
    
    return x

# ============================================================================
# STRATEGY 1: PURE GP (F1, F2, F3)
# ============================================================================

def pure_gp_strategy(X, y, dims, maximize=True):
    """Standard GP + Expected Improvement"""
    # Train GP
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*dims, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                   normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    # Generate candidates
    candidates = np.random.uniform(0, 1, (1000, dims))
    
    # Compute EI
    y_best = y.max() if maximize else y.min()
    ei = expected_improvement(candidates, gp, y_best, maximize=maximize)
    
    # Select best
    best_idx = np.argmax(ei)
    next_query = candidates[best_idx]
    
    return next_query, gp, ei[best_idx]

# ============================================================================
# STRATEGY 2: GP + SVM SCREENING (F4, F6, F7)
# ============================================================================

def gp_svm_strategy(X, y, dims, maximize=True):
    """GP with SVM candidate screening"""
    # Train GP
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*dims, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                   normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    # SVM screening
    threshold = np.percentile(y, 70)
    y_binary = (y > threshold).astype(int) if maximize else (y < threshold).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_scaled, y_binary)
    
    # Generate many candidates
    candidates = np.random.uniform(0, 1, (5000, dims))
    candidates_scaled = scaler.transform(candidates)
    
    # SVM screening: Keep top 1000 by probability
    probs = svm.predict_proba(candidates_scaled)[:, 1]
    top_indices = np.argsort(probs)[-1000:]
    filtered_candidates = candidates[top_indices]
    
    # EI on filtered
    y_best = y.max() if maximize else y.min()
    ei = expected_improvement(filtered_candidates, gp, y_best, maximize=maximize)
    
    best_idx = np.argmax(ei)
    next_query = filtered_candidates[best_idx]
    
    return next_query, gp, ei[best_idx]

# ============================================================================
# STRATEGY 3: GP + GRADIENTS (F5)
# ============================================================================

def gp_gradient_strategy(X, y, dims, maximize=True):
    """GP with gradient-guided refinement"""
    # Train GP
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*dims, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                   normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    # Multi-start gradient ascent
    best_f = -np.inf if maximize else np.inf
    best_x = None
    
    for restart in range(10):
        x_start = np.random.uniform(0, 1, dims)
        x_opt = gradient_ascent(gp, x_start, maximize=maximize)
        f_opt = gp.predict(x_opt.reshape(1, -1))[0]
        
        if (maximize and f_opt > best_f) or (not maximize and f_opt < best_f):
            best_f = f_opt
            best_x = x_opt
    
    return best_x, gp, best_f

# ============================================================================
# STRATEGY 4: FULL HYBRID (F8)
# ============================================================================

def full_hybrid_strategy(X, y, dims, maximize=True):
    """SVM + GP + Gradient optimization"""
    # Train GP
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*dims, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                   normalize_y=True, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    # SVM screening
    threshold = np.percentile(y, 60)  # Less strict for small data
    y_binary = (y > threshold).astype(int) if maximize else (y < threshold).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(kernel='rbf', C=0.5, gamma='scale', probability=True)
    svm.fit(X_scaled, y_binary)
    
    # Generate many candidates
    candidates = np.random.uniform(0, 1, (10000, dims))
    candidates_scaled = scaler.transform(candidates)
    
    # SVM screening
    probs = svm.predict_proba(candidates_scaled)[:, 1]
    top_indices = np.argsort(probs)[-500:]
    filtered_candidates = candidates[top_indices]
    
    # Gradient optimization on top candidates
    best_f = -np.inf if maximize else np.inf
    best_x = None
    
    for i in range(min(10, len(filtered_candidates))):
        x_start = filtered_candidates[i]
        x_opt = gradient_ascent(gp, x_start, maximize=maximize, max_steps=50)
        f_opt = gp.predict(x_opt.reshape(1, -1))[0]
        
        if (maximize and f_opt > best_f) or (not maximize and f_opt < best_f):
            best_f = f_opt
            best_x = x_opt
    
    return best_x, gp, best_f

# ============================================================================
# RUN ALL FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("NEXT QUERY RECOMMENDATIONS")
print("="*80)

recommendations = {}

# F1, F2, F3: Pure GP
for func_name in ['F1', 'F2', 'F3']:
    X, y, goal, dims = all_data[func_name]
    maximize = (goal == 'MAXIMIZE')
    next_query, gp, score = pure_gp_strategy(X, y, dims, maximize)
    recommendations[func_name] = {
        'query': next_query,
        'strategy': 'Pure GP',
        'score': score,
        'current_best': y.max() if maximize else y.min()
    }
    print(f"\n{func_name} (Pure GP):")
    print(f"   Next query: {next_query}")
    print(f"   EI score: {score:.6f}")
    print(f"   Current best: {recommendations[func_name]['current_best']:.6f}")

# F4, F6, F7: GP + SVM
for func_name in ['F4', 'F6', 'F7']:
    X, y, goal, dims = all_data[func_name]
    maximize = (goal == 'MAXIMIZE')
    next_query, gp, score = gp_svm_strategy(X, y, dims, maximize)
    recommendations[func_name] = {
        'query': next_query,
        'strategy': 'GP + SVM',
        'score': score,
        'current_best': y.max() if maximize else y.min()
    }
    print(f"\n{func_name} (GP + SVM):")
    print(f"   Next query: {next_query}")
    print(f"   EI score: {score:.6f}")
    print(f"   Current best: {recommendations[func_name]['current_best']:.6f}")

# F5: GP + Gradients
X, y, goal, dims = all_data['F5']
next_query, gp, score = gp_gradient_strategy(X, y, dims, maximize=True)
recommendations['F5'] = {
    'query': next_query,
    'strategy': 'GP + Gradients',
    'score': score,
    'current_best': y.max()
}
print(f"\nF5 (GP + Gradients):")
print(f"   Next query: {next_query}")
print(f"   Predicted value: {score:.6f}")
print(f"   Current best: {y.max():.6f}")

# F8: Full Hybrid
X, y, goal, dims = all_data['F8']
next_query, gp, score = full_hybrid_strategy(X, y, dims, maximize=True)
recommendations['F8'] = {
    'query': next_query,
    'strategy': 'Full Hybrid',
    'score': score,
    'current_best': y.max()
}
print(f"\nF8 (Full Hybrid - SVM + GP + Gradients):")
print(f"   Next query: {next_query}")
print(f"   Predicted value: {score:.6f}")
print(f"   Current best: {y.max():.6f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n{'Function':<10} {'Strategy':<20} {'Next Query'}")
print("-"*80)
for func_name in ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']:
    rec = recommendations[func_name]
    query_str = np.array2string(rec['query'], precision=4, separator=', ')
    print(f"{func_name:<10} {rec['strategy']:<20} {query_str}")

print("\n✅ All recommendations generated!")
print("\nCopy these queries for Week 6 submission.")
