import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
model_name = "google/vit-base-patch16-224-in21k"
extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=4)

# Use absolute path (edit to your exact folder)
model_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\models\vit_lung_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

model.to(device)
model.eval()
print("Model loaded successfully ✅")

# Validation data
val_path = r"E:\KIIT\Job\Group_Projects\5_Minor_Project\Data\valid"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=extractor.image_mean, std=extractor.image_std)
])
val_data = datasets.ImageFolder(val_path, transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)

# Predict
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nEvaluation Report:\n")
print(classification_report(y_true, y_pred, target_names=val_data.classes, digits=4))
