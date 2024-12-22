from data.data_processing import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

is_test = True

X, y = process_data(is_test)

X_test_tensor = torch.tensor(X, dtype=torch.float32)

class StepPredictor(nn.Module):
    def __init__(self, input_size):
        super(StepPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

input_size = X.shape[1]
model = StepPredictor(input_size)
model.load_state_dict(torch.load('best_model.pth'))#fine_tuned_model
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor).squeeze().numpy() 

results = pd.DataFrame({
    "True Steps": y,
    "Predicted Steps": predictions
})
print(results.head(20))  # Display first 20 rows

plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.6, label="Predictions")
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Perfect Prediction")
plt.xlabel("True Steps")
plt.ylabel("Predicted Steps")
plt.title("True vs Predicted Steps")
plt.legend()
plt.grid(True)
plt.show()

mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")