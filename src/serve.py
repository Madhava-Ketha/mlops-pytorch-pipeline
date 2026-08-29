import io
import os
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image
import torch
import torch.nn.functional as F
from src.dataset import get_transforms
from src.model import get_model

app = FastAPI(title="CIFAR-10 Classification Service")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms(train=False)


def load_trained_model():
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/app/checkpoints")
    model_name = os.getenv("MODEL_NAME", "classifier_v1.pt")
    model_path = Path(checkpoint_dir) / model_name

    if not model_path.exists():
        return False

    try:
        loaded_model = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load(str(model_path), map_location=device)
        if "model_state_dict" in checkpoint:
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            loaded_model.load_state_dict(checkpoint)
        loaded_model.to(device)
        loaded_model.eval()
        model = loaded_model
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
def startup_event():
    load_trained_model()


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if model is None:
        if not load_trained_model():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not ready or checkpoint not found."
            )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> Dict:
    if model is None and not load_trained_model():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        class_idx = predicted_idx.item()
        return {
            "prediction": class_idx,
            "class_name": CLASSES[class_idx],
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                CLASSES[i]: round(probabilities[i].item(), 4) for i in range(len(CLASSES))
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(exc)}")