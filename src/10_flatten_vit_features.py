import pandas as pd
import os

# ==============================
# PATHS
# ==============================
vit_input_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_vit.csv"
vit_output_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\vit_features_avg.csv"

# ==============================
# LOAD CSV
# ==============================
print("📂 Loading ViT features...")
vit_df = pd.read_csv(vit_input_csv)
print(f"✅ Loaded ViT features: {vit_df.shape}")

# ==============================
# COMPUTE ROW-WISE AVERAGE
# ==============================
if 'label' not in vit_df.columns:
    raise ValueError("❌ 'label' column not found in ViT CSV!")

vit_labels = vit_df['label']
vit_features = vit_df.drop(columns=['label'])
vit_avg = vit_features.mean(axis=1)

# Save averaged results
vit_avg_df = pd.DataFrame({
    'vit_avg': vit_avg,
    'label': vit_labels
})

os.makedirs(os.path.dirname(vit_output_csv), exist_ok=True)
vit_avg_df.to_csv(vit_output_csv, index=False)

print(f"✅ ViT averaged features saved to: {vit_output_csv}")
print(vit_avg_df.head())
