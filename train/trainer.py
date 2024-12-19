import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import RatFinderDataset
from models.time_step_predictor import TimeStepPredictor
from train.config import TrainingConfig

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create dataset
        dataset = RatFinderDataset(config.data_dir)
        
        # Split into train and validation sets
        self.train_dataset, self.val_dataset = dataset.split_data(train_ratio=0.8)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )
        
        # Initialize model
        self.model = TimeStepPredictor()
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # Track best validation loss
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Run one epoch of training"""
        self.model.train()  # Set model to training mode
        total_loss = 0
        
        for prob_grids, blocked_ratios, true_remaining_steps in self.train_loader:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_steps = self.model(prob_grids, blocked_ratios)
            
            # Compute loss
            loss = self.criterion(
                predicted_steps.squeeze(), 
                true_remaining_steps.squeeze()
            )
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        
        with torch.no_grad():  # No need to track gradients
            for prob_grids, blocked_ratios, true_remaining_steps in self.val_loader:
                # Forward pass
                predicted_steps = self.model(prob_grids, blocked_ratios)
                
                # Compute loss
                loss = self.criterion(
                    predicted_steps.squeeze(), 
                    true_remaining_steps.squeeze()
                )
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss = self.validate()
            
            # Print epoch statistics
            print(f'Epoch [{epoch+1}/{self.config.num_epochs}]')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(), 
                    self.config.model_save_path
                )
                print(f'New best model saved!')
            
            print('-' * 50)