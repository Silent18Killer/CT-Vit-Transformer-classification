import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==============================
# PATHS
# ==============================
csv_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\fused_avg_features.csv"
save_dir = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_svm"

# ==============================
# LOAD DATA
# ==============================
print("📂 Loading fused averaged features...")
df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset: {df.shape}")

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
# FEATURE SCALING
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# TRAIN SVM CLASSIFIER
# ==============================
print("\n🚀 Training SVM model...")
svm_model = SVC(
    kernel="rbf",        # You can also try: 'linear', 'poly', 'sigmoid'
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)
svm_model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Final SVM Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:")
report = classification_report(y_test, y_pred, digits=4)
print(report)

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"SVM Confusion Matrix (Accuracy: {acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")

os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "confusion_matrix_svm.png"))
plt.close()

# ==============================
# SAVE MODEL & REPORT
# ==============================
import joblib
joblib.dump(svm_model, os.path.join(save_dir, "svm_fused_model.pkl"))

with open(os.path.join(save_dir, "report.txt"), "w") as f:
    f.write(f"SVM Final Accuracy: {acc*100:.2f}%\n\n")
    f.write(report)

print(f"\n📁 Results saved to → {save_dir}")
print("✅ SVM classification completed successfully!")
