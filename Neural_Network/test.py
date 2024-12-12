import numpy as np

filename = "seed_457.npz"

data = np.load(filename)
print("Keys in the .npz file:", data.files)

# Access and print specific data arrays
for key in data.files:
    print(f"\nKey: {key}")
    print("Data shape:", data[key].shape)
    print("Sample data:", data[key][0])  # Print the first entry of each array