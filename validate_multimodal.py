import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torchvision import models
import glob
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = ASLDataset('dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Model definition
class MultiModalNet(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.cnn = models.mobilenet_v2(pretrained=False)
        self.cnn.classifier = nn.Identity()
        cnn_out_dim = 1280
        self.landmark_fc = nn.Sequential(
            nn.Linear(21*3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
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

model = MultiModalNet(num_classes=24).to(device)
model.load_state_dict(torch.load('asl_multimodal.pth', map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, landmarks, labels in dataloader:
        images = images.to(device)
        landmarks = landmarks.to(device)
        outputs = model(images, landmarks)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Metrics
print("Classification Report:")
alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
print(classification_report(all_labels, all_preds, target_names=alphabet))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=alphabet, yticklabels=alphabet, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
