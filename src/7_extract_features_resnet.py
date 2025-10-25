import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
data_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data"  # contains train & valid
save_csv = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\features\features_resnet.csv"

# ==============================
# DEVICE
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using Device: {device}")

# ==============================
# TRANSFORMS (same as ResNet training)
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# LOAD DATA FROM BOTH TRAIN & VALID
# ==============================
dataset = datasets.ImageFolder(data_path, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

class_names = dataset.classes
print(f"Classes: {class_names}")

# ==============================
# LOAD PRETRAINED RESNET50
# ==============================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.to(device)

# Remove final FC → We take last feature layer output
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# ==============================
# EXTRACT FEATURES
# ==============================
feature_list = []
label_list = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc="Extracting Features"):
        imgs = imgs.to(device)

        # Output shape: (batch, 2048, 1, 1)
        feats = model(imgs)
        feats = feats.view(feats.size(0), -1)  # Flatten → (batch, 2048)

        feature_list.append(feats.cpu().numpy())
        label_list.append(labels.numpy())

# Stack all batches together
features = np.vstack(feature_list)
labels = np.hstack(label_list)

print("✅ Features Shape:", features.shape)
print("✅ Labels Shape:", labels.shape)

# ==============================
# SAVE TO CSV
# ==============================
os.makedirs(os.path.dirname(save_csv), exist_ok=True)

df = pd.DataFrame(features)
df['label'] = labels
df.to_csv(save_csv, index=False)

print(f"📁 Features saved successfully → {save_csv}")