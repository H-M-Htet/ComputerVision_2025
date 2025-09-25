import numpy as np
X = np.load("calibration.npz")
print("mtx shape:", X["mtx"].shape)
print("dist shape:", X["dist"].shape)
