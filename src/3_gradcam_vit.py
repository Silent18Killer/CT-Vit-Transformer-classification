import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Load fine-tuned ViT model
# ==========================================
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=4)
model_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\models\vit_lung_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()
model.set_attn_implementation("eager")  # needed for output_attentions=True
print("✅ Model loaded successfully!")

# ==========================================
# Load image processor
# ==========================================
processor = ViTImageProcessor.from_pretrained(model_name)

# ==========================================
# Pick one image
# ==========================================
img_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data\valid\adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib\000109 (3).png"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

image = Image.open(img_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

# ==========================================
# Forward pass with attention outputs
# ==========================================
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# ==========================================
# Extract last layer attention map
# ==========================================
attentions = outputs.attentions[-1]  # [batch, heads, tokens, tokens]
# Average across heads
att_map = attentions.mean(1)[0]  # [tokens, tokens]

# We only need attention of CLS token to image patches
cls_att = att_map[0, 1:]  # drop the CLS token itself
cls_att = cls_att.reshape(14, 14).cpu().numpy()  # 196 -> 14x14

# ==========================================
# Create Grad-CAM style overlay
# ==========================================
cls_att = (cls_att - cls_att.min()) / (cls_att.max() - cls_att.min())
cls_att = cv2.resize(cls_att, (image.size[0], image.size[1]))

heatmap = cv2.applyColorMap(np.uint8(255 * cls_att), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

# ==========================================
# Show & Save Result
# ==========================================
plt.imshow(overlay)
plt.axis("off")
plt.title("ViT Focus Area (Grad-CAM-like Visualization)")
plt.show()

save_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\results\gradcam_overlay.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"✅ Grad-CAM saved at: {save_path}")
