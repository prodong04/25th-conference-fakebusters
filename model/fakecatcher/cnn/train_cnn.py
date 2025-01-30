import torch
from torch.utils.data import DataLoader, random_split
from cnn.classifier import Model
import yaml
import json
import numpy as np
from torch.utils.data import Dataset
import argparse
import logging

# Argument parsing
parser = argparse.ArgumentParser(description='Train CNN model for fakecatcher')
parser.add_argument('-c', '--config_path', required=True, help='Path to the config.yaml file')
parser.add_argument('-i', '--input_path', required=True, help='Path to the ppg_map_results_updated.json file')
parser.add_argument('-l', '--log_path', required=False, help='Path to save logs')
parser.add_argument('-o', '--output_path', required=False, help='Path to save the trained model')
args = parser.parse_args()

# Logging setup
log_path = args.log_path if args.log_path else 'train_cnn.log'
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started training process')

# 설정
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f'Using device: {device}')

# 데이터 준비
logging.info(f'Loading configuration from {args.config_path}')
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)
    fps_standard = config.get('fps_standard', 30)  # 기본값 30
    time_interval = config.get('seg_time_interval', 1)  # 기본값 1
    w = fps_standard * time_interval

logging.info(f'Loading data from {args.input_path}')
with open(args.input_path, 'r') as file:
    data = json.load(file)

# Extract PPG Maps and Labels
ppg_maps = []
labels = []
for item in data:
    ppg_maps.append(np.array(item['ppg_map']))  # (64, w) 형태의 배열
    labels.append(item['label'])               # 0 or 1

# Convert to numpy arrays
ppg_maps = np.array(ppg_maps)
labels = np.array(labels)     

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (np.array): Shape (2831, 64, 90)
            labels (np.array): Shape (2831,)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (data_sample, label)
        """
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(ppg_maps, labels)
logging.info(f"Dataset size: {len(dataset)}")
logging.info(f"Sample PPG map shape: {dataset[0][0].shape}")
logging.info(f"Sample label: {dataset[0][1]}")
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
logging.info(f"Training dataset size: {train_size}")
logging.info(f"Validation dataset size: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 준비
model_name = 'ResNet34'  # "ResNet34", "EfficientNetB3"로 변경 가능
logging.info(f'Initializing model: {model_name}')
model = Model.get_model(model_name, w)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습
logging.info('Starting training')
Model.train(model, train_loader, val_loader, criterion, optimizer, device, epochs=NUM_EPOCHS)
logging.info('Training completed')

# Save the model if output path is provided
if args.output_path:
    torch.save(model.state_dict(), args.output_path)
    logging.info(f'Model saved to {args.output_path}')