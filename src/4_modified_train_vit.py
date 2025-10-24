import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

# ==============================
# Configuration
# ==============================
train_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data\train"
val_path   = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data\valid"
model_dir  = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\models"
result_dir = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_final"

num_classes = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/vit-base-patch16-224-in21k"

print(f"✅ Using device: {device}")

# ==============================
# Model & Preprocessing
# ==============================
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ==============================
# Load Datasets
# ==============================
if not os.path.exists(train_path):
    raise FileNotFoundError(f"❌ Training folder not found: {train_path}")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"❌ Validation folder not found: {val_path}")

train_data = datasets.ImageFolder(train_path, transform=transform)
val_data   = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=8)

print("✅ Data loaded successfully!")
print(f"Classes found: {train_data.classes}")

# ==============================
# Training Setup
# ==============================
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()
epochs = 50
patience = 5
best_val_acc = 0
no_improve = 0
train_losses, val_losses, val_accuracies = [], [], []

# Create directories
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Create log file
log_file = os.path.join(result_dir, "training_log.txt")
with open(log_file, "w") as f:
    f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\n")

# ==============================
# Training Loop with Early Stopping
# ==============================
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

    avg_train_loss = total_loss / len(train_loader)

    # ==============================
    # Validation
    # ==============================
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).logits
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}: "
          f"Train Loss={avg_train_loss:.4f}, "
          f"Val Loss={avg_val_loss:.4f}, "
          f"Val Acc={val_acc*100:.2f}%")

    # Save to log file
    with open(log_file, "a") as f:
        f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc*100:.2f}\n")

    # ==============================
    # Early Stopping & Model Saving
    # ==============================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_path = os.path.join(model_dir, f"vit_best_model_{timestamp}.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Model improved! Saved at epoch {epoch+1} ({best_val_acc*100:.2f}%)")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"🛑 Early stopping triggered at epoch {epoch+1}")
        print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
        break

# ==============================
# Final Evaluation
# ==============================
model.load_state_dict(torch.load(best_model_path))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ==============================
# Metrics Report
# ==============================
report = classification_report(y_true, y_pred, target_names=val_data.classes, digits=4)
acc = np.mean(np.array(y_true) == np.array(y_pred))
print("\nClassification Report:")
print(report)
print(f"Final Validation Accuracy: {acc*100:.2f}%")

metrics_file = os.path.join(result_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(report)
    f.write(f"\nFinal Accuracy: {acc*100:.2f}%")

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_data.classes, yticklabels=val_data.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))

# ==============================
# Plot Loss & Accuracy Curves
# ==============================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Val Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(result_dir, "loss_curve.png"))
plt.close()

plt.figure(figsize=(8,5))
plt.plot(np.array(val_accuracies)*100, label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.savefig(os.path.join(result_dir, "val_accuracy_curve.png"))
plt.close()

print(f"\n✅ Training complete!")
print(f"Best Model saved at: {best_model_path}")
print(f"All metrics and plots saved to: {result_dir}")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
