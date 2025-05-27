import os
import torch
import requests
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

LABEL_MAP = {
    1: "Frackels",
    2: "Acne scars",
    3: "Mole and Tags",
    4: "Acne"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Your Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1WPu5NMKlyVMx5uvs8GYKGECoiqTJ90Mm"
MODEL_PATH = "multi_condition_detector.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("✅ Model downloaded")

def load_model(model_path, num_classes):
    download_model()
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model(MODEL_PATH, num_classes=len(LABEL_MAP) + 1)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    resized_image = image.resize((224, 224))
    image_tensor = transform(resized_image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    results = []
    for i, score in enumerate(predictions["scores"]):
        label_idx = predictions["labels"][i].item()
        label_name = LABEL_MAP.get(label_idx, "Unknown")

        if (label_name == "Mole and Tags" and score < 0.2) or (label_name != "Mole and Tags" and score < 0.1):
            continue

        box = predictions["boxes"][i].tolist()
        scale_x = img_width / 224
        scale_y = img_height / 224
        scaled_box = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ]
        normalized_box = [
            scaled_box[0] / img_width,
            scaled_box[1] / img_height,
            scaled_box[2] / img_width,
            scaled_box[3] / img_height
        ]
        results.append({
            "label": label_name,
            "score": round(score.item(), 2),
            "box": normalized_box
        })

    return results
