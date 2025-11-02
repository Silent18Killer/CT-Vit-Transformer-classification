import pandas as pd
import os

# ==============================
# PATHS
# ==============================
resnet_avg_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\resnet_features_avg.csv"
vit_avg_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\vit_features_avg.csv"
fused_output_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\fused_avg_features.csv"

# ==============================
# LOAD BOTH
# ==============================
resnet_df = pd.read_csv(resnet_avg_csv)
vit_df = pd.read_csv(vit_avg_csv)

if not all(resnet_df['label'] == vit_df['label']):
    raise ValueError("❌ Label mismatch between ResNet and ViT CSVs!")

# ==============================
# FUSION (AVERAGE OF BOTH)
# ==============================
fused_avg = (resnet_df['resnet_avg'] + vit_df['vit_avg']) / 2
fused_df = pd.DataFrame({
    'fused_avg': fused_avg,
    'label': resnet_df['label']
})

os.makedirs(os.path.dirname(fused_output_csv), exist_ok=True)
fused_df.to_csv(fused_output_csv, index=False)

print(f"✅ Fused features saved successfully → {fused_output_csv}")
print(fused_df.head())
