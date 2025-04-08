import os
import glob
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Prepare landmark dataset with augmentation
X = []
y = []
root_dir = 'dataset_cropped'
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}

for cls_name, idx in class_to_idx.items():
    cls_dir = os.path.join(root_dir, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for npy_path in glob.glob(os.path.join(cls_dir, '*.npy')):
        landmarks = np.load(npy_path)
        # Data augmentation: add small noise
        for _ in range(5):  # generate 5 augmented samples per original
            noise = np.random.normal(0, 0.01, size=landmarks.shape)
            augmented = landmarks + noise
            X.append(augmented)
            y.append(idx)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} augmented samples with {X.shape[1]} features each.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Train SVM ---
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred))
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
joblib.dump(svm_pipeline, 'asl_landmark_svm_augmented.joblib')
print("Saved SVM model to asl_landmark_svm_augmented.joblib")

# --- Train CNN ---
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 21, 3)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = LandmarkDataset(X_train, y_train)
test_dataset = LandmarkDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LandmarkCNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = LandmarkCNN(num_classes=24).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)

# Train CNN
cnn_model.train()
for epoch in range(20):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/20 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

# Evaluate CNN
cnn_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = cnn_model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("CNN Classification Report:")
print(classification_report(all_labels, all_preds))
torch.save(cnn_model.state_dict(), 'asl_landmark_cnn_augmented.pth')
print("Saved CNN model to asl_landmark_cnn_augmented.pth")
