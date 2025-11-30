from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Recapture Detection API")

# Allow local testing from browsers/tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Model loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path: str = "recapture_detection_model.pth"):
    try:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


try:
    model = load_model()
except FileNotFoundError:
    model = None
    # App still starts; endpoints will return meaningful error if model missing
except Exception as e:
    raise


# Preprocessing (must match training)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_NAMES = ["fake", "real"]


def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    contents = await file.read()
    try:
        img = read_imagefile(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][pred_idx].item())

    return JSONResponse({
        "pred_index": pred_idx,
        "pred_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
    })


# Serve static frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<html><body><h3>Index not found.</h3></body></html>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
