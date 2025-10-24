import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# Configurations
# ==============================
train_path = "../Data/train"
val_path   = "../Data/valid"
num_classes = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/vit-base-patch16-224-in21k"

# ==============================
# Model & Preprocessing
# ==============================
extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
])

train_data = datasets.ImageFolder(train_path, transform=transform)
val_data   = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=8)

# ==============================
# Training
# ==============================
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs).logits
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# ==============================
# Evaluation
# ==============================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ==============================
# Metrics
# ==============================
print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=val_data.classes, digits=4)
print(report)
acc = np.mean(np.array(y_true)==np.array(y_pred))
print(f"Accuracy: {acc*100:.2f}%")

# Save results
os.makedirs("../results", exist_ok=True)
with open("../results/metrics.txt", "w") as f:
    f.write(report)
    f.write(f"\nAccuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_data.classes, yticklabels=val_data.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("../results/confusion_matrix.png")

# Save model
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/vit_lung_model.pth")
print("\nModel saved successfully.")
