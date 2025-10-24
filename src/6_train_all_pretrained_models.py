import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, SwinForImageClassification, AutoImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os, datetime

# ==============================
# Configuration
# ==============================
train_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data\train"
val_path   = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data\valid"
base_model_dir = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\models_all"
base_result_dir = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_all"

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 4
epochs = 15
batch_size = 8
patience = 3

os.makedirs(base_model_dir, exist_ok=True)
os.makedirs(base_result_dir, exist_ok=True)

print(f"✅ Using device: {device}")

# ==============================
# Common Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_path, transform=transform)
val_data   = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size)

print("✅ Dataset Loaded Successfully!")
print(f"Classes: {train_data.classes}")

# ==============================
# Define All Models
# ==============================
def get_model(name):
    if name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "vit":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=num_classes, ignore_mismatched_sizes=True)
    elif name == "swin":
        processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        model = SwinForImageClassification.from_pretrained(
            "microsoft/swin-base-patch4-window7-224", num_labels=num_classes, ignore_mismatched_sizes=True)
    else:
        raise ValueError("❌ Unknown model name")
    return model.to(device)

# ==============================
# Training & Evaluation Function
# ==============================
def train_and_evaluate(model_name):
    print(f"\n🚀 Training {model_name.upper()} model...")

    model = get_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc, no_improve = 0, 0
    train_losses, val_losses, val_accuracies = [], [], []

    model_dir = os.path.join(base_model_dir, model_name)
    result_dir = os.path.join(base_result_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    log_file = os.path.join(result_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).logits if hasattr(model, "logits") else model(imgs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).logits if hasattr(model, "logits") else model(imgs)
                if isinstance(outputs, tuple): outputs = outputs[0]
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

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc*100:.2f}%")
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc*100:.2f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"{model_name}_best_{timestamp}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model improved! Saved at epoch {epoch+1} ({best_val_acc*100:.2f}%)")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"🛑 Early stopping at epoch {epoch+1}. Best acc: {best_val_acc*100:.2f}%")
            break

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).logits if hasattr(model, "logits") else model(imgs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=val_data.classes, digits=4)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n📊 Final Accuracy ({model_name.upper()}): {acc*100:.2f}%\n")
    print(report)

    # Save report
    with open(os.path.join(result_dir, "metrics.txt"), "w") as f:
        f.write(report)
        f.write(f"\nFinal Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=val_data.classes, yticklabels=val_data.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    # Plot Loss & Accuracy Curves
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.title(f"Loss Curve - {model_name.upper()}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig(os.path.join(result_dir, "loss_curve.png")); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(np.array(val_accuracies)*100, label="Val Accuracy", color="green")
    plt.title(f"Accuracy Curve - {model_name.upper()}")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.savefig(os.path.join(result_dir, "accuracy_curve.png")); plt.close()

    return acc

# ==============================
# Run All Models
# ==============================
model_list = ["vgg16", "resnet50", "densenet121", "efficientnet_b0", "vit", "swin"]
results_summary = {}

for name in model_list:
    acc = train_and_evaluate(name)
    results_summary[name] = acc

# ==============================
# Save Summary
# ==============================
summary_file = os.path.join(base_result_dir, "all_model_summary.txt")
with open(summary_file, "w") as f:
    f.write("Model\tAccuracy(%)\n")
    for k, v in results_summary.items():
        f.write(f"{k}\t{v*100:.2f}\n")

print("\n🏁 All models trained and results saved!")
for k, v in results_summary.items():
    print(f"{k}: {v*100:.2f}%")
print(f"\n📁 Summary saved at: {summary_file}")
