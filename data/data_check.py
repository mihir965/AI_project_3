import numpy as np

data = np.load("data/seed_457_8649.npz")
for key in data.files:
    print(f"{key}: shape={data[key].shape}, sample={data[key][0]}")
