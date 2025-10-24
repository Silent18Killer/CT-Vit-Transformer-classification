import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import os

# ==============================
# Configuration
# ==============================
data_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data"
num_classes = 4
num_epochs = 5        # you can increase to 20–30 later
batch_size = 8
k_folds = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/vit-base-patch16-224-in21k"

print(f"✅ Using device: {device}")

# ==============================
# Preprocessing
# ==============================
processor = ViTImageProcessor.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)
print(f"✅ Dataset loaded: {len(dataset)} samples, Classes: {dataset.classes}")

# ==============================
# K-Fold Setup
# ==============================
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []

# ==============================
# K-Fold Training
# ==============================
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"\n========== Fold {fold+1}/{k_folds} ==========")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    # Model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    for epoch in range(num_epochs):
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
        print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss={total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, target_names=dataset.classes)
    print(f"Fold {fold+1} Accuracy: {acc*100:.2f}%")

    fold_results.append({
        "fold": fold+1,
        "accuracy": acc,
        "precision": np.mean([report[c]["precision"] for c in dataset.classes]),
        "recall": np.mean([report[c]["recall"] for c in dataset.classes]),
        "f1": np.mean([report[c]["f1-score"] for c in dataset.classes]),
    })

# ==============================
# Average Metrics
# ==============================
avg_acc = np.mean([r["accuracy"] for r in fold_results])
avg_prec = np.mean([r["precision"] for r in fold_results])
avg_rec = np.mean([r["recall"] for r in fold_results])
avg_f1 = np.mean([r["f1"] for r in fold_results])

print("\n========== K-Fold Cross Validation Results ==========")
for r in fold_results:
    print(f"Fold {r['fold']}: "
          f"Acc={r['accuracy']*100:.2f}%, "
          f"Prec={r['precision']:.4f}, "
          f"Recall={r['recall']:.4f}, "
          f"F1={r['f1']:.4f}")

print("\n✅ Average Across All Folds:")
print(f"Accuracy: {avg_acc*100:.2f}%")
print(f"Precision: {avg_prec:.4f}")
print(f"Recall: {avg_rec:.4f}")
print(f"F1-score: {avg_f1:.4f}")

# Save results
os.makedirs(r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_kfold", exist_ok=True)
with open(r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_kfold\kfold_metrics.txt", "w") as f:
    for r in fold_results:
        f.write(f"Fold {r['fold']}: "
                f"Acc={r['accuracy']*100:.2f}%, "
                f"Prec={r['precision']:.4f}, "
                f"Recall={r['recall']:.4f}, "
                f"F1={r['f1']:.4f}\n")
    f.write("\nAverage:\n")
    f.write(f"Accuracy: {avg_acc*100:.2f}%\n")
    f.write(f"Precision: {avg_prec:.4f}\n")
    f.write(f"Recall: {avg_rec:.4f}\n")
    f.write(f"F1-score: {avg_f1:.4f}\n")

print("\n📁 All results saved in /results_kfold folder")
