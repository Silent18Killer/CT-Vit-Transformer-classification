import torch
from transformers import ViTModel, ViTImageProcessor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
data_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data"  # your dataset folder
save_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_vit.csv"

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using Device: {device}")

# ==============================
# LOAD TRANSFORMER PROCESSOR
# ==============================
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ==============================
# LOAD ENTIRE DATASET
# ==============================
dataset = datasets.ImageFolder(data_path, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

class_names = dataset.classes
print(f"Classes: {class_names}")

# ==============================
# LOAD PRETRAINED ViT MODEL
# ==============================
model = ViTModel.from_pretrained(model_name)
model.to(device)
model.eval()

# ==============================
# EXTRACT FEATURES
# ==============================
feature_list = []
label_list = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc="Extracting ViT Features"):
        imgs = imgs.to(device)

        outputs = model(imgs)
        # CLS token = Global representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        feature_list.append(cls_embeddings.cpu().numpy())
        label_list.append(labels.numpy())

# Join all batches
features = np.vstack(feature_list)  # shape: (N, 768)
labels = np.hstack(label_list)

print("✅ Global Feature Vector Shape:", features.shape)
print("✅ Labels Shape:", labels.shape)

# ==============================
# SAVE TO CSV
# ==============================
os.makedirs(os.path.dirname(save_csv), exist_ok=True)

df = pd.DataFrame(features)
df['label'] = labels
df.to_csv(save_csv, index=False)

print(f"📁 Global Features saved successfully → {save_csv}")
