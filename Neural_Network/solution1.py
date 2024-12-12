import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error,
    r2_score
)

class RatCatchingDataset(Dataset):
    def __init__(self, bot_grid, rat_grid, time_rem):
        self.bot_grid = torch.tensor(bot_grid, dtype=torch.float32)
        self.rat_grid = torch.tensor(rat_grid, dtype=torch.float32)
        self.time_rem = torch.tensor(time_rem, dtype=torch.float32)
        
    def __len__(self):
        return len(self.time_rem)
    
    def __getitem__(self, idx):
        input_tensor = torch.stack([
            self.bot_grid[idx], 
            self.rat_grid[idx]
        ])
        return input_tensor, self.time_rem[idx]

class RatCatchingNetwork(nn.Module):
    def __init__(self, grid_size=30):
        super(RatCatchingNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.flattened_size = self._get_conv_output_size(grid_size)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def _get_conv_output_size(self, grid_size):
        test_input = torch.zeros(1, 2, grid_size, grid_size)
        output = self.conv_layers(test_input)
        return output.view(1, -1).size(1)
    
    def forward(self, x):
        conv_features = self.conv_layers(x)
        flattened = conv_features.view(conv_features.size(0), -1)
        output = self.fc_layers(flattened)
        return output.squeeze()

def evaluate_model_performance(model_path, bot_grid, rat_grid, time_rem):
    # Load the data
    dataset = RatCatchingDataset(bot_grid, rat_grid, time_rem)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    model = RatCatchingNetwork()
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to store predictions and true values
    predictions = []
    true_values = []
    
    # Disable gradient computation
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Make predictions
            outputs = model(inputs)
            
            # Convert to numpy for metric calculation
            predictions.extend(outputs.numpy())
            true_values.extend(targets.numpy())
    
    # Calculate various regression metrics
    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(true_values, predictions),
        'Mean Squared Error (MSE)': mean_squared_error(true_values, predictions),
        'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(true_values, predictions)),
        'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(true_values, predictions),
        'R2 Score': r2_score(true_values, predictions)
    }
    
    return metrics

def main():
    # Load data
    data = np.load('seed_457.npz')
    bot_grid_array = data['bot_grid']
    rat_grid_array = data['rat_grid']
    time_rem_array = data['time_step_remaining']
    
    # Evaluate model performance
    performance_metrics = evaluate_model_performance(
        'best_model.pth', 
        bot_grid_array, 
        rat_grid_array, 
        time_rem_array
    )
    
    # Print metrics
    print("Model Performance Metrics:")
    for metric_name, metric_value in performance_metrics.items():
        print(f"{metric_name}: {metric_value}")

if __name__ == '__main__':
    main()