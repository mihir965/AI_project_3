from train.trainer import Trainer
from train.config import TrainingConfig

def main():
    # Create training configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()