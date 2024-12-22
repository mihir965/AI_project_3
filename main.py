from data.data_processing import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = process_data(False)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create dataset and loader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the neural network
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

# Initialize model
input_size = X.shape[1]
model = StepPredictor(input_size)

# Loss and optimizer
criterion = nn.MSELoss()  # Regression loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Track losses
train_losses = []
val_losses = []

# Early stopping parameters
best_val_loss = float('inf')
patience = 10
wait = 0

epochs = 100
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_outputs = model(val_X).squeeze()
            val_loss = criterion(val_outputs, val_y)
            total_val_loss += val_loss.item()
    
    # Compute average losses
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Store losses for plotting
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Print loss values
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered!")
            break

# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()