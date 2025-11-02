import pandas as pd
import numpy as np
import os

# ==============================
# PATHS
# ==============================
resnet_input_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_resnet.csv"
resnet_output_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\resnet_features_avg.csv"

# ==============================
# LOAD CSV
# ==============================
print("📂 Loading ResNet features...")
resnet_df = pd.read_csv(resnet_input_csv)
print(f"✅ Loaded ResNet features: {resnet_df.shape}")

# ==============================
# COMPUTE ROW-WISE AVERAGE
# ==============================
if 'label' not in resnet_df.columns:
    raise ValueError("❌ 'label' column not found in ResNet CSV!")

resnet_labels = resnet_df['label']
resnet_features = resnet_df.drop(columns=['label'])
resnet_avg = resnet_features.mean(axis=1)

# Save averaged results
resnet_avg_df = pd.DataFrame({
    'resnet_avg': resnet_avg,
    'label': resnet_labels
})

os.makedirs(os.path.dirname(resnet_output_csv), exist_ok=True)
resnet_avg_df.to_csv(resnet_output_csv, index=False)

print(f"✅ ResNet averaged features saved to: {resnet_output_csv}")
print(resnet_avg_df.head())
