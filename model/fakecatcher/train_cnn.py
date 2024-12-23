import torch
from torch.utils.data import DataLoader, random_split
from cnn.classifier import Model

# 설정
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 준비

dataset = 
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 준비
model_name = 'baseCNN'  # "ResNet34", "EfficientNetB3"로 변경 가능
model = Model.get_model(model_name)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습
Model.train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)

# # 테스트
# test_loss, test_accuracy = Model.test(model, val_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
