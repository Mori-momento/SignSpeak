import os
import numpy as np
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import glob

# Parameters
data_dir = 'dataset'
num_classes = 24
batch_size = 16
num_epochs = 10
learning_rate = 1e-3
output_model_path = 'asl_multimodal.pth'

# Custom dataset
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class MultiModalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Image branch
        self.cnn = models.mobilenet_v2(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Remove classifier
        cnn_out_dim = 1280
        # Landmark branch
        self.landmark_fc = nn.Sequential(
            nn.Linear(21*3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, landmarks):
        img_feat = self.cnn(image)
        lm_feat = self.landmark_fc(landmarks)
        combined = torch.cat((img_feat, lm_feat), dim=1)
        out = self.classifier(combined)
        return out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ASLDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = MultiModalNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, landmarks, labels in dataloader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    torch.save(model.state_dict(), output_model_path)
    print(f"Multi-modal model saved to {output_model_path}")
