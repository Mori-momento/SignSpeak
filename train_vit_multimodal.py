import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob
import timm

# Parameters
data_dir = 'dataset_cropped'
num_classes = 24
batch_size = 16
num_epochs = 30
learning_rate = 1e-3
weight_decay = 1e-4
patience = 5
output_model_path = 'asl_vit_multimodal.pth'

# Aggressive augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.RandomAffine(20, translate=(0.2,0.2), scale=(0.8,1.2), shear=10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    ], p=0.8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        for cls_name, idx in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_path in glob.glob(os.path.join(cls_dir, '*.jpg')):
                npy_path = img_path.replace('.jpg', '.npy')
                if os.path.exists(npy_path):
                    self.samples.append((img_path, npy_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, npy_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        landmarks = np.load(npy_path)
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        return image, landmarks, label

class ViTMultiModal(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        vit_out_dim = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.landmark_fc = nn.Sequential(
            nn.Linear(21*3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(vit_out_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, landmarks):
        img_feat = self.vit(image)
        lm_feat = self.landmark_fc(landmarks)
        combined = torch.cat((img_feat, lm_feat), dim=1)
        out = self.classifier(combined)
        return out

if __name__ == '__main__':
    dataset_full = ASLDataset(data_dir, transform=transform)

    # Train/val split
    val_ratio = 0.2
    val_size = int(len(dataset_full) * val_ratio)
    train_size = len(dataset_full) - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ViTMultiModal(num_classes).to(device)

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
        for images, landmarks, labels in train_loader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            for images, landmarks, labels in val_loader:
                images = images.to(device)
                landmarks = landmarks.to(device)
                labels = labels.to(device)
                outputs = model(images, landmarks)
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
