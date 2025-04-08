import os
import torch
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Parameters
data_dir = 'dataset'
num_classes = 24
batch_size = 16
num_epochs = 30
learning_rate = 1e-3
weight_decay = 1e-4
patience = 5
output_model_path = 'asl_mobilenetv2_augmented.pth'

# Aggressive data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomAffine(20, translate=(0.2,0.2), scale=(0.8,1.2), shear=10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    ], p=0.8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    dataset_full = datasets.ImageFolder(data_dir, transform=transform)

    # Train/val split
    val_ratio = 0.2
    val_size = int(len(dataset_full) * val_ratio)
    train_size = len(dataset_full) - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    # Replace classifier with dropout
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_model_path)
            print(f"Saved best model to {output_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
