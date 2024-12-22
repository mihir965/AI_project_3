from data.data_processing import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class StepPredictor(nn.Module):
    def __init__(self, input_size):
        super(StepPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),   # Input layer
            nn.ReLU(),                   # Activation
            nn.Linear(128, 64),          # Hidden layer
            nn.ReLU(),                   # Activation
            nn.Linear(64, 32),           # Hidden layer
            nn.ReLU(),                   # Activation
            nn.Linear(32, 1)             # Output layer (single value for regression)
        )
        
    def forward(self, x):
        return self.model(x)

X, y = process_data(False)

input_size = X.shape[1]  # Make sure input size matches your dataset
model = StepPredictor(input_size)
model.load_state_dict(torch.load('best_model.pth'))  # Load saved weights
model.train()

# Loss and optimizer
criterion = nn.MSELoss()  # Regression loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import DataLoader, TensorDataset

# Prepare training data
X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
epochs = 50  # Add more epochs as needed
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), 'fine_tuned_model.pth')
