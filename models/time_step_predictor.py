import torch
import torch.nn as nn

class TimeStepPredictor(nn.Module):
    def __init__(self, grid_size=30):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # Reduces size to 15x15
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)   # Reduces size to 7x7
        )
        
        # Calculate flattened conv output size
        # After two MaxPool2d layers, spatial dimensions are reduced by factor of 4
        # Final output will be 64 channels * 7 * 7
        conv_output_size = 64 * 7 * 7  # This is the correct size calculation
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + 1, 128),  # +1 for blocked ratio
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, prob_grids, blocked_ratio):
        # Print shapes for debugging
        # print(f"prob_grids shape: {prob_grids.shape}")
        # print(f"blocked_ratio shape: {blocked_ratio.shape}")
        
        conv_output = self.conv_layers(prob_grids)
        # print(f"conv_output shape: {conv_output.shape}")
        
        flattened = torch.flatten(conv_output, start_dim=1)
        # print(f"flattened shape: {flattened.shape}")
        
        combined = torch.cat([flattened, blocked_ratio], dim=1)
        # print(f"combined shape: {combined.shape}")
        
        return self.fc_layers(combined)