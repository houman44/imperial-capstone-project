import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist

# Base paths - UPDATE THIS TO YOUR ACTUAL PATH
base_path = Path("/Users/houman/Downloads/initial_data")

# Verify path exists
if not base_path.exists():
    print(f"ERROR: Path does not exist: {base_path}")
    print("Please update the base_path variable to match your directory structure")
    import sys
    sys.exit(1)

print(f"Loading data from: {base_path}\n")

# Analyze all 8 functions
results = []

for func_num in range(1, 9):
    func_dir = base_path / f"function_{func_num}"
    X_file = func_dir / "initial_inputs.npy"
    y_file = func_dir / "initial_outputs.npy"
    
    try:
        X = np.load(X_file)
        y = np.load(y_file)
        
        # Calculate metrics
        n_samples = X.shape[0]
        n_dims = X.shape[1] if X.ndim > 1 else 1
        
        # Output characteristics
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        cv = y_std / y_mean if y_mean != 0 else np.inf  # Coefficient of variation
        
        # Input space coverage (average pairwise distance)
        if n_dims > 1:
            from scipy.spatial.distance import pdist
            pairwise_dists = pdist(X)
            mean_dist = np.mean(pairwise_dists)
            std_dist = np.std(pairwise_dists)
        else:
            pairwise_dists = pdist(X.reshape(-1, 1))
            mean_dist = np.mean(pairwise_dists)
            std_dist = np.std(pairwise_dists)
        
        # Sample efficiency: samples per dimension
        samples_per_dim = n_samples / n_dims
        
        # Difficulty score (heuristic)
        # Higher dimensionality, lower samples/dim, higher variance = harder
        difficulty = (
            n_dims * 2 +  # Dimensionality penalty
            (1 / samples_per_dim) * 50 +  # Sample efficiency penalty
            cv * 10 +  # Variance penalty
            (1 / (y_range + 0.01)) * 5  # Flat landscape penalty
        )
        
        results.append({
            'function': func_num,
            'n_samples': n_samples,
            'n_dims': n_dims,
            'samples_per_dim': samples_per_dim,
            'y_mean': y_mean,
            'y_std': y_std,
            'y_min': y_min,
            'y_max': y_max,
            'y_range': y_range,
            'cv': cv,
            'mean_dist': mean_dist,
            'difficulty': difficulty
        })
        
        print(f"\n{'='*60}")
        print(f"FUNCTION {func_num}")
        print(f"{'='*60}")
        print(f"Dimensions: {n_dims}D")
        print(f"Samples: {n_samples} ({samples_per_dim:.1f} per dimension)")
        print(f"Output range: [{y_min:.4f}, {y_max:.4f}] (range: {y_range:.4f})")
        print(f"Output stats: mean={y_mean:.4f}, std={y_std:.4f}, CV={cv:.4f}")
        print(f"Space coverage: mean_dist={mean_dist:.4f}, std_dist={std_dist:.4f}")
        print(f"Difficulty score: {difficulty:.2f}")
        
    except Exception as e:
        print(f"\nFunction {func_num}: Error - {e}")

# Sort by difficulty
results_sorted = sorted(results, key=lambda x: x['difficulty'], reverse=True)

print(f"\n\n{'='*60}")
print("RANKING BY DIFFICULTY (HARDEST TO EASIEST)")
print(f"{'='*60}")
print(f"{'Rank':<6} {'Func':<6} {'Dims':<6} {'Samples':<8} {'S/D':<8} {'Range':<10} {'CV':<8} {'Score':<8}")
print(f"{'-'*60}")

for rank, r in enumerate(results_sorted, 1):
    print(f"{rank:<6} {r['function']:<6} {r['n_dims']:<6} {r['n_samples']:<8} "
          f"{r['samples_per_dim']:<8.1f} {r['y_range']:<10.4f} {r['cv']:<8.3f} {r['difficulty']:<8.1f}")

# Detailed analysis of top 3 hardest
print(f"\n\n{'='*60}")
print("TOP 3 MOST CHALLENGING FUNCTIONS - DETAILED ANALYSIS")
print(f"{'='*60}")

for rank in range(min(3, len(results_sorted))):
    r = results_sorted[rank]
    print(f"\n#{rank+1}: FUNCTION {r['function']} (Score: {r['difficulty']:.1f})")
    print(f"  Why it's challenging:")
    
    if r['n_dims'] >= 6:
        print(f"    ⚠️  HIGH DIMENSIONALITY: {r['n_dims']}D (curse of dimensionality)")
    
    if r['samples_per_dim'] < 5:
        print(f"    ⚠️  SPARSE SAMPLING: Only {r['samples_per_dim']:.1f} samples/dimension")
    
    if r['cv'] > 0.5:
        print(f"    ⚠️  HIGH VARIANCE: CV={r['cv']:.3f} (noisy/multi-modal landscape)")
    
    if r['y_range'] < 0.2:
        print(f"    ⚠️  FLAT LANDSCAPE: Range={r['y_range']:.4f} (hard to find optima)")
    
    print(f"  Recommendations:")
    print(f"    - Use {max(50, r['n_dims'] * 10)} restarts")
    print(f"    - Set xi={0.05 + r['n_dims'] * 0.01:.3f} for more exploration")
    if r['n_dims'] >= 6:
        print(f"    - Consider dimensionality reduction or ARD kernel")
    if r['cv'] > 0.5:
        print(f"    - Add noise parameter to GP (alpha > 1e-6)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Difficulty scores
ax = axes[0, 0]
funcs = [r['function'] for r in results_sorted]
scores = [r['difficulty'] for r in results_sorted]
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(funcs)))
ax.barh(funcs, scores, color=colors)
ax.set_xlabel('Difficulty Score (higher = harder)')
ax.set_ylabel('Function')
ax.set_title('Optimization Difficulty Ranking')
ax.invert_yaxis()
for i, (f, s) in enumerate(zip(funcs, scores)):
    ax.text(s + 1, f, f'{s:.1f}', va='center')

# Plot 2: Dimensionality vs Samples/Dim
ax = axes[0, 1]
dims = [r['n_dims'] for r in results]
spd = [r['samples_per_dim'] for r in results]
funcs_all = [r['function'] for r in results]
scatter = ax.scatter(dims, spd, c=scores, s=200, cmap='RdYlGn_r', edgecolors='black', linewidths=2)
for f, d, s in zip(funcs_all, dims, spd):
    ax.annotate(f'F{f}', (d, s), ha='center', va='center', fontweight='bold')
ax.set_xlabel('Dimensions')
ax.set_ylabel('Samples per Dimension')
ax.set_title('Sample Efficiency vs Dimensionality')
ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Min recommended (5)')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Difficulty')

# Plot 3: Output characteristics
ax = axes[1, 0]
y_ranges = [r['y_range'] for r in results]
cvs = [r['cv'] for r in results]
scatter = ax.scatter(y_ranges, cvs, c=scores, s=200, cmap='RdYlGn_r', edgecolors='black', linewidths=2)
for f, yr, cv in zip(funcs_all, y_ranges, cvs):
    ax.annotate(f'F{f}', (yr, cv), ha='center', va='center', fontweight='bold')
ax.set_xlabel('Output Range')
ax.set_ylabel('Coefficient of Variation')
ax.set_title('Output Landscape Characteristics')
plt.colorbar(scatter, ax=ax, label='Difficulty')

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')
table_data = []
table_data.append(['Rank', 'Func', 'Dims', 'Samples', 'S/D', 'Difficulty'])
for rank, r in enumerate(results_sorted[:5], 1):
    table_data.append([
        f"#{rank}",
        f"F{r['function']}",
        f"{r['n_dims']}D",
        f"{r['n_samples']}",
        f"{r['samples_per_dim']:.1f}",
        f"{r['difficulty']:.1f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.1, 0.1, 0.15, 0.15, 0.15, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color by rank
colors_table = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, 5))
for i in range(1, 6):
    for j in range(6):
        table[(i, j)].set_facecolor(colors_table[i-1])

ax.set_title('Top 5 Most Challenging Functions', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('function_difficulty_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n\nVisualization saved as 'function_difficulty_analysis.png'")
plt.show()

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")
print(f"Most challenging: Function {results_sorted[0]['function']} "
      f"({results_sorted[0]['n_dims']}D, score: {results_sorted[0]['difficulty']:.1f})")
print(f"Least challenging: Function {results_sorted[-1]['function']} "
      f"({results_sorted[-1]['n_dims']}D, score: {results_sorted[-1]['difficulty']:.1f})")