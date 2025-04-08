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

def augment_landmarks(landmarks, num_aug=20):
    augmented = []
    for _ in range(num_aug):
        noise = np.random.normal(0, 0.01, size=landmarks.shape)
        scale = np.random.uniform(0.9, 1.1)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        lm = landmarks.reshape(-1,3).copy()
        # Rotate around wrist (assumed at index 0)
        wrist = lm[0]
        for i in range(1, 21):
            x, y = lm[i,0]-wrist[0], lm[i,1]-wrist[1]
            x_new = x*cos_a - y*sin_a
            y_new = x*sin_a + y*cos_a
            lm[i,0] = wrist[0] + scale*x_new
            lm[i,1] = wrist[1] + scale*y_new
            lm[i,2] = lm[i,2] * scale
        lm += noise.reshape(-1,3)
        augmented.append(lm.flatten())
    return augmented

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
        X.append(landmarks)
        y.append(idx)
        augmented = augment_landmarks(landmarks, num_aug=20)
        X.extend(augmented)
        y.extend([idx]*20)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} augmented samples with {X.shape[1]} features each.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Train SVM ---
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred))
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
joblib.dump(svm_pipeline, 'asl_landmark_svm_augmented_v2.joblib')
print("Saved SVM model to asl_landmark_svm_augmented_v2.joblib")

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
torch.save(cnn_model.state_dict(), 'asl_landmark_cnn_augmented_v2.pth')
print("Saved CNN model to asl_landmark_cnn_augmented_v2.pth")
