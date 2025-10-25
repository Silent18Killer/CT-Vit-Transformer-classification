import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os

# Paths
resnet_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_resnet.csv"
vit_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_vit.csv"
save_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\fused_avg_features.csv"

# Load both features
df_res = pd.read_csv(resnet_csv)
df_vit = pd.read_csv(vit_csv)

print("ResNet Shape:", df_res.shape)
print("ViT Shape:", df_vit.shape)

# Separate labels
labels = df_res['label']
df_res = df_res.drop(columns=['label'])
df_vit = df_vit.drop(columns=['label'])

# ✅ Reduce ResNet 2048 → 768 using PCA
pca = PCA(n_components=df_vit.shape[1])
resnet_reduced = pca.fit_transform(df_res)

print("✅ ResNet reduced shape:", resnet_reduced.shape)

# ✅ Feature Averaging
fused_features = (resnet_reduced + df_vit.values) / 2.0

# Create final dataframe
df_fused = pd.DataFrame(fused_features)
df_fused['label'] = labels

# Save
os.makedirs(os.path.dirname(save_csv), exist_ok=True)
df_fused.to_csv(save_csv, index=False)

print("✅ Averaged Feature Shape:", df_fused.shape)
print(f"📁 Saved Successfully → {save_csv}")
