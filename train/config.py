from dataclasses import dataclass

@dataclass
class TrainingConfig:
    data_dir: str = './data'
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    model_save_path: str = 'rat_finder_model.pth'
    train_ratio: float = 0.8  # Added parameter 