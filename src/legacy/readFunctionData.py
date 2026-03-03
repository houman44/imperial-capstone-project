import numpy as np
import pandas as pd

""" Load and print initial function data
#for i in range(1, 9):
    X1 = np.load(f'Downloads/initial_data/function_{i}/initial_inputs.npy')
    print(X1)
    y1 = np.load(f'Downloads/initial_data/function_{i}/initial_outputs.npy')
    print(y1)
    print(f'Function {i} data shapes: X1: {X1.shape}, y1: {y1.shape}')  
"""

X = np.loadtxt(f'Downloads/capWeek1Processed/inputs.txt')
print(X)
y = np.loadtxt(f'Downloads/capWeek1Processed/outputs.txt')
print(y)
z = pd.np.load('Downloads/capWeek1Processed/readFunctionData.npy')
z = np.load(f'Downloads/capWeek1Processed/readFunctionData.npy')
print(z)
print(f'Processed data shapes: X: {X.shape}, y: {y.shape}, z: {z.shape}')
