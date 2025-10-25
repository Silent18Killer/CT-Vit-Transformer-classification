import pandas as pd
import os

# Paths
resnet_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_resnet.csv"
vit_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_vit.csv"
save_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\fused_features.csv"

# Load both CSVs
df_res = pd.read_csv(resnet_csv)
df_vit = pd.read_csv(vit_csv)

print("✅ ResNet CSV Shape:", df_res.shape)
print("✅ ViT CSV Shape:", df_vit.shape)

# Check alignment
assert len(df_res) == len(df_vit), "❌ Feature rows don't match!"

# Remove label from ViT to avoid duplication
df_vit = df_vit.drop(columns=['label'])

# Horizontally fuse (concatenate)
df_fused = pd.concat([df_res, df_vit], axis=1)

# Save fused dataset
os.makedirs(os.path.dirname(save_csv), exist_ok=True)
df_fused.to_csv(save_csv, index=False)

print(f"✅ Fused Feature Shape: {df_fused.shape}")
print(f"📁 Fusion CSV saved → {save_csv}")
