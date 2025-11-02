import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import joblib

# ==============================
# PATHS
# ==============================
csv_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\fused_avg_features.csv"
save_dir = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results_xgboost"

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
# SCALING (optional for XGBoost)
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# TRAIN XGBOOST CLASSIFIER
# ==============================
print("\n🚀 Training XGBoost model...")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,        # number of boosting rounds
    learning_rate=0.05,      # step size shrinkage
    max_depth=5,             # depth of each tree
    subsample=0.8,           # training instance subsample ratio
    colsample_bytree=0.8,    # feature subsample ratio
    gamma=0.1,               # regularization term
    reg_lambda=1,            # L2 regularization
    random_state=42,
    n_jobs=-1,
    objective="multi:softmax",   # since it's a classification
    num_class=len(np.unique(y))  # number of classes
)

xgb_model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = xgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Final XGBoost Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:")
report = classification_report(y_test, y_pred, digits=4)
print(report)

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title(f"XGBoost Confusion Matrix (Accuracy: {acc*100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")

os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "confusion_matrix_xgboost.png"))
plt.close()

# ==============================
# SAVE MODEL & REPORT
# ==============================
joblib.dump(xgb_model, os.path.join(save_dir, "xgboost_fused_model.pkl"))

with open(os.path.join(save_dir, "report.txt"), "w") as f:
    f.write(f"XGBoost Final Accuracy: {acc*100:.2f}%\n\n")
    f.write(report)

print(f"\n📁 Results saved to → {save_dir}")
print("✅ XGBoost classification completed successfully!")
