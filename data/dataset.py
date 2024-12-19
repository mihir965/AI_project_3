import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class RatFinderDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ]
        
        # Preload and concatenate data
        self.bot_grids = []
        self.rat_grids = []
        self.time_steps_remaining = []
        self.blocked_ratios = []
        
        for file_path in self.data_files:
            data = np.load(file_path)
            
            self.bot_grids.append(data['bot_grid'])
            self.rat_grids.append(data['rat_grid'])
            self.time_steps_remaining.append(data['time_step_remaining'])
            self.blocked_ratios.append(data['blocked_ratio'])
        
        # Concatenate all data
        self.bot_grids = np.concatenate(self.bot_grids)
        self.rat_grids = np.concatenate(self.rat_grids)
        self.time_steps_remaining = np.concatenate(self.time_steps_remaining)
        self.blocked_ratios = np.concatenate(self.blocked_ratios)
    
    def split_data(self, train_ratio=0.8):
        """
        Split dataset into training and validation sets
        
        Args:
            train_ratio (float): Ratio of data to use for training (0 to 1)
        """
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            self, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )  
        return train_dataset, val_dataset
    
    def __len__(self):
        return len(self.time_steps_remaining)
    
    def __getitem__(self, idx):
        # Stack bot and rat probability grids
        prob_grids = np.stack([
            self.bot_grids[idx], 
            self.rat_grids[idx]
        ])
        
        # Ensure correct shapes
        prob_grids = torch.FloatTensor(prob_grids)  # Shape: [2, 30, 30]
        blocked_ratio = torch.FloatTensor([self.blocked_ratios[idx]])  # Shape: [1]
        time_steps = torch.FloatTensor([self.time_steps_remaining[idx]])  # Shape: [1]
        
        return prob_grids, blocked_ratio, time_steps