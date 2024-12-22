from .basecnn import baseCNN
from .resnet import ResNet34
from .effinet import EfficientNetB3
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:
    @staticmethod
    def get_model(model_name):
        if model_name == 'baseCNN':
            return baseCNN()
        elif model_name == 'ResNet34':
            return ResNet34()
        elif model_name == 'EfficientNetB3':
            return EfficientNetB3()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
        model.to(device)
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = correct / len(val_loader.dataset)

            # Print training and validation results
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')

    @staticmethod
    def test(model, dataloader, criterion, device):
        model.to(device)  # 모델을 디바이스로 이동
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)  # 레이블을 (batch_size, 1)로 변경
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (outputs > 0.5).float()  # 0.5 임계값을 기준으로 예측
                correct += (preds == labels).sum().item()
        accuracy = correct / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy
