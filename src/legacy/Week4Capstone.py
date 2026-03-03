
"""
Boundary points:
- [0.00, 1.00, 1.00, 1.00] → 4440.5 (BEST - at boundary)
- [0.00, 1.00, 1.00, 0.95] → 3819.7 (14% worse)

These are EXACTLY support vectors:
- Lie on the decision boundary (x4 = 1.0 vs x4 = 0.95)
- Define the "good" region (x2, x3, x4 all → 1.0)
- Small movement (0.05) causes large change (620 units)
- Margin width ≈ 0.05 in x4 dimension

"""