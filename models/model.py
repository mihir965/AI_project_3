import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class GridModel(nn.Module):
    def __init__(self):
        super(GridModel, self).__init__()

        # CNN path for prob grid 1
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten()
        )

        self.fc_scaler = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(1568 * 2 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, grid1, grid2, scalar):
        #CNN outputs
        out1 = self.cnn1(grid1)
        out2 = self.cnn2(grid2)
        #Scalar output
        out_scalar = self.fc_scaler(scalar)
        combined = torch.cat((out1, out2, out_scalar), dim=1)
        return self.combined_fc(combined)
    
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for grid1, grid2, scalar, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(grid1, grid2, scalar)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# Testing Function
def test_model(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for grid1, grid2, scalar, labels in dataloader:
            outputs = model(grid1, grid2, scalar)
            predictions.append(outputs.numpy())
            actuals.append(labels.numpy())
    return predictions, actuals


# Load Data from NPZ File
def load_data(filename):
    data = np.load(filename)
    bot_grid = torch.tensor(data['bot_grid']).unsqueeze(1).float()
    rat_grid = torch.tensor(data['rat_grid']).unsqueeze(1).float()
    scalar_features = torch.tensor(np.column_stack((
        data['time_step_remaining'],
        data['blocked_ratio'],
        data['time_step'],
        data['dist_to_target'],
        data['target_cell_prob'],
        data['max_prob']
    ))).float()
    labels = torch.tensor(data['time_step_remaining']).float().unsqueeze(1)

    dataset = TensorDataset(bot_grid, rat_grid, scalar_features, labels)
    return dataset


# Example Usage
if __name__ == '__main__':
    # Initialize model
    model = GridModel()
    load_model(model, 'grid_model.pth')

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 32

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load Dataset
    dataset = load_data('./data/seed_457_6766`.npz')  # Replace 'data.npz' with your file
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the Model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

    # Save the Model
    save_model(model, 'grid_model_updated.pth')

    # Load and Test the Model
    load_model(model, 'grid_model_updated.pth')
    predictions, actuals = test_model(model, dataloader)

    print("Testing completed.")

    # Concatenate and flatten the predictions and actuals
    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()

    # Save predictions and actuals to a file
    np.savez_compressed(
        'test_results.npz',
        predictions=predictions,
        actuals=actuals
    )

    # Print some sample results
    print("Sample Predictions vs Actual Values:")
    for i in range(10):  # Display first 10 results
        print(f"Prediction: {predictions[i]:.4f}, Actual: {actuals[i]:.4f}")
