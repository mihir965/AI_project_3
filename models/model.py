import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Model Definition
class GridModel(nn.Module):
    def __init__(self):
        super(GridModel, self).__init__()

        # CNN layer for grid 1
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # CNN layer for grid 2
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # Fully connected layer for scalar features
        self.fc_scaler = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combined layers
        self.combined_fc = nn.Sequential(
            nn.Linear(1568 * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, grid1, grid2, scalar):
        out1 = self.cnn1(grid1)
        out2 = self.cnn2(grid2)
        out_scalar = self.fc_scaler(scalar)
        combined = torch.cat((out1, out2, out_scalar), dim=1)
        return self.combined_fc(combined)

# Function to load the npz files
def load_data(filename):
    data = np.load(filename)
    bot_grid = torch.tensor(data['bot_grid']).unsqueeze(1).float()
    rat_grid = torch.tensor(data['rat_grid']).unsqueeze(1).float()

    # Normalize scalar features
    scaler = StandardScaler()
    scalar_features = torch.tensor(scaler.fit_transform(np.column_stack((
        data['time_step_remaining'],
        data['blocked_ratio'],
        data['time_step'],
        data['dist_to_target'],
        data['target_cell_prob'],
        data['max_prob']
    ))), dtype=torch.float32)

    labels = torch.tensor(data['time_step_remaining']).float().unsqueeze(1)
    return TensorDataset(bot_grid, rat_grid, scalar_features, labels)

# This will train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for grid1, grid2, scalar, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(grid1, grid2, scalar)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # This puts the model into evaluation mode for validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for grid1, grid2, scalar, labels in val_loader:
                outputs = model(grid1, grid2, scalar)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss / len(val_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# Testing Model with new data
def test_model(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for grid1, grid2, scalar, labels in dataloader:
            outputs = model(grid1, grid2, scalar)
            predictions.append(outputs.numpy())
            actuals.append(labels.numpy())

    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"Test Results - MAE: {mae:.4f}, R²: {r2:.4f}")

    results_df = pd.DataFrame({'Prediction': predictions, 'Actual': actuals})
    results_df.to_csv('test_results.csv', index=False)

    print("Sample Predictions vs Actual Values:")
    print(results_df.head(10))

    return predictions, actuals

# Save and Load Model
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

if __name__ == '__main__':
    model = GridModel()
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 64

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    dataset = load_data('./data/seed_457_35249.npz')
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

    # save_model(model, 'grid_model_updated.pth')
    load_model(model, 'grid_model_updated.pth')
    predictions, actuals = test_model(model, val_loader)
