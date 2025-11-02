import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================
# PATHS
# ==============================
csv_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\fused_avg_features.csv"
save_dir = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_fcnn"

# ==============================
# LOAD DATA
# ==============================
print("📂 Loading dataset...")
df = pd.read_csv(csv_path)
print("✅ Loaded:", df.shape)

# Separate features and labels
X = df.drop(columns=["label"]).values
y = df["label"].values

# ==============================
# TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ==============================
# MODEL DEFINITION
# ==============================
input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

class FCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

model = FCNN(input_dim, num_classes)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ==============================
# TRAINING SETUP
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
batch_size = 32

# ==============================
# TRAINING LOOP
# ==============================
print("\n🚀 Training started...")
train_losses, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    total_loss = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = torch.argmax(outputs, dim=1)
        acc = (preds.cpu() == y_test).float().mean().item()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(X_train):.4f} | Val Acc: {acc*100:.2f}%")
    train_losses.append(total_loss / len(X_train))
    val_accuracies.append(acc)

# ==============================
# FINAL EVALUATION
# ==============================
model.eval()
with torch.no_grad():
    outputs = model(X_test.to(device))
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# Accuracy
accuracy = accuracy_score(y_test, preds)
print(f"\n✅ Final Accuracy: {accuracy*100:.2f}%")

# ==============================
# CLASSIFICATION REPORT
# ==============================
report = classification_report(y_test, preds, digits=4)
print("\n📊 Classification Report:")
print(report)

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"FCNN Confusion Matrix (Accuracy: {accuracy*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")

os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "confusion_matrix_fcnn.png"))
plt.close()

# ==============================
# SAVE MODEL, METRICS & PLOTS
# ==============================
torch.save(model.state_dict(), os.path.join(save_dir, "fcnn_fused_model.pth"))
with open(os.path.join(save_dir, "report.txt"), "w") as f:
    f.write(f"Final Accuracy: {accuracy*100:.2f}%\n\n")
    f.write(report)

# Plot Accuracy Curve
plt.figure(figsize=(7,4))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy over Epochs")
plt.legend()
plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
plt.close()

print(f"\n📁 All results saved in → {save_dir}")
print("✅ Done!")
