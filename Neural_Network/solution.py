import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score

class RatCatchingDataset(Dataset):
    def __init__(self, bot_grid, rat_grid, time_rem):
        # Convert numpy arrays to torch tensors
        self.bot_grid = torch.tensor(bot_grid, dtype=torch.float32)
        self.rat_grid = torch.tensor(rat_grid, dtype=torch.float32)
        self.time_rem = torch.tensor(time_rem, dtype=torch.float32)
        
    def __len__(self):
        return len(self.time_rem)
    
    def __getitem__(self, idx):
        # Combine bot and rat grids as input
        input_tensor = torch.stack([
            self.bot_grid[idx], 
            self.rat_grid[idx]
        ])
        return input_tensor, self.time_rem[idx]

class RatCatchingNetwork(nn.Module):
    def __init__(self, grid_size=30):
        super(RatCatchingNetwork, self).__init__()
        
        # Convolutional layers to extract features
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
        
        # Calculate the flattened size after convolutions
        self.flattened_size = self._get_conv_output_size(grid_size)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Single output for time steps remaining
        )
    
    def _get_conv_output_size(self, grid_size):
        # Simulate a forward pass to calculate flattened size
        test_input = torch.zeros(1, 2, grid_size, grid_size)
        output = self.conv_layers(test_input)
        return output.view(1, -1).size(1)
    
    def forward(self, x):
        # Convolutional feature extraction
        conv_features = self.conv_layers(x)
        
        # Flatten the features
        flattened = conv_features.view(conv_features.size(0), -1)
        
        # Fully connected layers
        output = self.fc_layers(flattened)
        
        return output.squeeze()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    # Training loop with validation
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

def calculate_r2_score(model_path, bot_grid, rat_grid, time_rem):
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
            
            # Convert to numpy for R2 score calculation
            predictions.extend(outputs.numpy())
            true_values.extend(targets.numpy())
    
    # Calculate R2 score
    r2 = r2_score(true_values, predictions)
    
    return r2

# def main():
#     # Load data
#     data = np.load('seed_457.npz')
#     bot_grid_array = data['bot_grid']
#     rat_grid_array = data['rat_grid']
#     time_rem_array = data['time_step_remaining']
    
#     # Create dataset
#     dataset = RatCatchingDataset(bot_grid_array, rat_grid_array, time_rem_array)
    
#     # Split into train and validation sets
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
#     # Create data loaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32)
    
#     # Initialize model
#     model = RatCatchingNetwork()
    
#     # Loss and optimizer
#     criterion = nn.MSELoss()  # Mean Squared Error for regression
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Train the model
#     train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)
    
#     # Optional: Plot training progress
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.title('Model Training Progress')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# if __name__ == '__main__':
#     main()

def main():
    # Load data
    data = np.load('seed_457.npz')
    bot_grid_array = data['bot_grid']
    rat_grid_array = data['rat_grid']
    time_rem_array = data['time_step_remaining']
    
    # Calculate R2 score
    r2_score = calculate_r2_score('best_model.pth', bot_grid_array, rat_grid_array, time_rem_array)
    
    print(f"R2 Score: {r2_score}")

if __name__ == '__main__':
    main()