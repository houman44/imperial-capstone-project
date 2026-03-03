"""
How to Actually Find Support Vectors in Bayesian Optimization Data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ============================================================================
# FUNCTION 5 DATA (Chemical Process)
# ============================================================================

X5 = np.array([
    [0.19, 0.04, 0.61, 0.41],  # → 64.4
    [0.76, 0.54, 0.66, 0.36],  # → 18.3
    [0.44, 0.80, 0.21, 0.15],  # → 0.11
    [0.35, 0.32, 0.12, 0.47],  # → 109.6
    [0.30, 0.32, 0.62, 0.10],  # → 63.4
    [0.57, 0.03, 0.88, 0.82],  # → 258.4
    [0.57, 0.18, 0.34, 0.79],  # → 8.85
    [0.83, 0.79, 0.11, 0.69],  # → 4.21
    [0.61, 0.84, 0.98, 0.21],  # → 28.3
    [0.04, 0.29, 0.98, 0.56],  # → 55.5
    [0.88, 0.29, 0.41, 0.03],  # → 0.51
    [0.96, 0.03, 0.09, 0.95],  # → 18.9
    [0.93, 0.85, 0.64, 0.59],  # → 113.7
    [0.85, 0.94, 0.48, 0.76],  # → 356.9
    [0.99, 0.90, 0.91, 0.33],  # → 67.1
    [0.30, 0.29, 0.46, 0.93],  # → 78.5
    [0.90, 0.55, 0.95, 0.48],  # → 432.9
    [0.50, 0.66, 0.31, 0.26],  # → 18.2
    [0.06, 0.58, 0.76, 0.66],  # → 44.3
    [0.67, 0.71, 0.77, 0.86],  # → 257.8
    [0.22, 0.85, 0.88, 0.88],  # → 1088.9
    [0.26, 0.84, 0.95, 0.89],  # → 1550.9
    [0.00, 1.00, 1.00, 1.00],  # → 4440.5 (BEST)
    [0.00, 1.00, 1.00, 0.95]   # → 3819.7
])

y5 = np.array([64.4, 18.3, 0.11, 109.6, 63.4, 258.4, 8.85, 4.21, 28.3, 55.5,
               0.51, 18.9, 113.7, 356.9, 67.1, 78.5, 432.9, 18.2, 44.3, 257.8,
               1088.9, 1550.9, 4440.5, 3819.7])

# ============================================================================
# METHOD 1: FIND SUPPORT VECTORS USING SVM
# ============================================================================

print("="*80)
print("METHOD 1: SUPPORT VECTORS FROM SVM CLASSIFICATION")
print("="*80)

# Step 1: Create binary labels (good vs bad)
threshold = np.percentile(y5, 70)  # Top 30% is "good"
y_binary = (y5 > threshold).astype(int)

print(f"\nThreshold: {threshold:.2f}")
print(f"Good points (1): {np.sum(y_binary == 1)}")
print(f"Bad points (0): {np.sum(y_binary == 0)}")

# Step 2: Standardize features (important for SVM!)
scaler = StandardScaler()
X5_scaled = scaler.fit_transform(X5)

# Step 3: Train SVM (with probability=True for predict_proba)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X5_scaled, y_binary)

# Step 4: Extract support vectors
support_vector_indices = svm.support_
support_vectors = X5[support_vector_indices]
support_vector_labels = y_binary[support_vector_indices]
support_vector_values = y5[support_vector_indices]

print(f"\n{'='*80}")
print(f"SUPPORT VECTORS FOUND: {len(support_vector_indices)}")
print(f"{'='*80}")

for i, idx in enumerate(support_vector_indices):
    sv = support_vectors[i]
    label = "GOOD" if support_vector_labels[i] == 1 else "BAD"
    value = support_vector_values[i]
    
    print(f"\nSV {i+1} (Index {idx}):")
    print(f"  Point: [{sv[0]:.2f}, {sv[1]:.2f}, {sv[2]:.2f}, {sv[3]:.2f}]")
    print(f"  Value: {value:.2f}")
    print(f"  Class: {label}")
    print(f"  Distance to boundary: {abs(svm.decision_function(X5_scaled[idx:idx+1])[0]):.4f}")

# ============================================================================
# METHOD 2: FIND BOUNDARY POINTS (High Uncertainty)
# ============================================================================

print(f"\n{'='*80}")
print("METHOD 2: BOUNDARY POINTS (Near Decision Boundary)")
print("="*80)

# Points where P(good) ≈ 0.5 are on the boundary
probabilities = svm.predict_proba(X5_scaled)[:, 1]  # P(good)
boundary_mask = np.abs(probabilities - 0.5) < 0.2  # Within 0.2 of 0.5

boundary_indices = np.where(boundary_mask)[0]
boundary_points = X5[boundary_indices]
boundary_probs = probabilities[boundary_indices]
boundary_values = y5[boundary_indices]

print(f"\nBoundary points found: {len(boundary_indices)}")
print(f"\nThese points are near the 'good' vs 'bad' decision boundary:\n")

for i, idx in enumerate(boundary_indices):
    point = boundary_points[i]
    prob = boundary_probs[i]
    value = boundary_values[i]
    
    print(f"Point {idx}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}, {point[3]:.2f}]")
    print(f"  Value: {value:.2f}, P(good): {prob:.3f}")

# ============================================================================
# METHOD 3: GRADIENT-BASED SUPPORT IDENTIFICATION
# ============================================================================

print(f"\n{'='*80}")
print("METHOD 3: HIGH-GRADIENT REGIONS (Rapid Change)")
print("="*80)

# Fit regression model to find gradients
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

kernel = C(1.0) * RBF(length_scale=0.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
gp.fit(X5, y5)

# Compute gradients numerically
def compute_gradient_magnitude(gp, x, epsilon=0.01):
    """Compute ||∇f(x)|| numerically"""
    gradients = []
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        
        grad_i = (gp.predict([x_plus])[0] - gp.predict([x_minus])[0]) / (2 * epsilon)
        gradients.append(grad_i)
    
    return np.linalg.norm(gradients)

gradient_magnitudes = np.array([compute_gradient_magnitude(gp, x) for x in X5])

# Points with high gradients are near boundaries
high_gradient_threshold = np.percentile(gradient_magnitudes, 75)
high_gradient_mask = gradient_magnitudes > high_gradient_threshold

high_gradient_indices = np.where(high_gradient_mask)[0]

print(f"\nPoints with high gradients (top 25%): {len(high_gradient_indices)}")
print(f"Gradient threshold: {high_gradient_threshold:.2f}\n")

for idx in high_gradient_indices:
    point = X5[idx]
    grad_mag = gradient_magnitudes[idx]
    value = y5[idx]
    
    print(f"Point {idx}: [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}, {point[3]:.2f}]")
    print(f"  Value: {value:.2f}, Gradient magnitude: {grad_mag:.2f}")

# ============================================================================
# VISUALIZE: 2D SLICE (x1=0, x4=1)
# ============================================================================

print(f"\n{'='*80}")
print("VISUALIZATION: 2D Slice (x1=0, x4=1)")
print("="*80)

# Filter points where x1 ≈ 0 and x4 ≈ 1
slice_mask = (X5[:, 0] < 0.1) & (X5[:, 3] > 0.9)
X_slice = X5[slice_mask]
y_slice = y5[slice_mask]

if len(X_slice) > 3:
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Data points
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_slice[:, 1], X_slice[:, 2], c=y_slice, 
                         s=200, cmap='viridis', edgecolors='black', linewidth=2)
    plt.colorbar(scatter, label='Function Value')
    plt.xlabel('x₂')
    plt.ylabel('x₃')
    plt.title('Function Values (x₁≈0, x₄≈1)')
    plt.grid(True, alpha=0.3)
    
    # Highlight support vectors in this slice
    for idx in support_vector_indices:
        if slice_mask[idx]:
            plt.scatter(X5[idx, 1], X5[idx, 2], 
                       marker='*', s=500, c='red', 
                       edgecolors='black', linewidth=2,
                       label='Support Vector' if idx == support_vector_indices[0] else '')
    
    plt.legend()
    
    # Plot 2: Decision boundary
    plt.subplot(1, 2, 2)
    
    # Create grid
    x2_range = np.linspace(0, 1, 100)
    x3_range = np.linspace(0, 1, 100)
    X2, X3 = np.meshgrid(x2_range, x3_range)
    
    # Create 4D points with x1=0, x4=1
    grid_points = np.c_[np.zeros(10000), X2.ravel(), X3.ravel(), np.ones(10000)]
    grid_scaled = scaler.transform(grid_points)
    
    # Predict probabilities
    Z = svm.predict_proba(grid_scaled)[:, 1].reshape(100, 100)
    
    # Plot decision boundary
    contour = plt.contourf(X2, X3, Z, levels=20, cmap='RdYlGn', alpha=0.6)
    plt.colorbar(contour, label='P(good)')
    plt.contour(X2, X3, Z, levels=[0.5], colors='black', linewidths=3)
    
    # Plot actual points
    good_mask = y_slice > threshold
    plt.scatter(X_slice[good_mask, 1], X_slice[good_mask, 2], 
               c='green', s=200, marker='o', edgecolors='black', 
               linewidth=2, label='Good', alpha=0.8)
    plt.scatter(X_slice[~good_mask, 1], X_slice[~good_mask, 2], 
               c='red', s=200, marker='o', edgecolors='black', 
               linewidth=2, label='Bad', alpha=0.8)
    
    # Highlight support vectors
    for idx in support_vector_indices:
        if slice_mask[idx]:
            plt.scatter(X5[idx, 1], X5[idx, 2], 
                       marker='*', s=500, c='yellow', 
                       edgecolors='black', linewidth=2)
    
    plt.xlabel('x₂')
    plt.ylabel('x₃')
    plt.title('SVM Decision Boundary (Black line = P=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/support_vectors_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to support_vectors_visualization.png")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY: THREE WAYS TO IDENTIFY CRITICAL POINTS")
print("="*80)

print("\n1. SVM SUPPORT VECTORS:")
print(f"   - {len(support_vector_indices)} points that define the decision boundary")
print(f"   - These are the ONLY points needed to classify new data")
print(f"   - Mathematically: Points where αᵢ > 0 in SVM optimization")

print("\n2. BOUNDARY POINTS:")
print(f"   - {len(boundary_indices)} points near P(good) = 0.5")
print(f"   - High uncertainty about classification")
print(f"   - Most informative for active learning")

print("\n3. HIGH-GRADIENT POINTS:")
print(f"   - {len(high_gradient_indices)} points with large ||∇f(x)||")
print(f"   - Regions where function changes rapidly")
print(f"   - Useful for gradient-based optimization")

print("\n" + "="*80)
print("INTERSECTION ANALYSIS")
print("="*80)

# Check overlap
sv_set = set(support_vector_indices)
boundary_set = set(boundary_indices)
gradient_set = set(high_gradient_indices)

sv_and_boundary = sv_set & boundary_set
sv_and_gradient = sv_set & gradient_set
all_three = sv_set & boundary_set & gradient_set

print(f"\nSupport vectors that are ALSO boundary points: {len(sv_and_boundary)}")
print(f"Support vectors that are ALSO high-gradient: {len(sv_and_gradient)}")
print(f"Points that are ALL THREE: {len(all_three)}")

if all_three:
    print("\nMost critical points (all three methods agree):")
    for idx in all_three:
        print(f"  Index {idx}: {X5[idx]} → {y5[idx]:.2f}")