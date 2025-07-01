import torch
import faiss
import yaml
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
# from torchvision import transforms
import uvicorn
import pickle

from src.model import ViTContrastive
from src.dataset import val_transforms

app = FastAPI()

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ViTContrastive(embed_dim=768, projection_dim=128, pretrained=False)
model.load_state_dict(torch.load(cfg['model_path'], map_location=device))
model.to(device)
model.eval()

# Load Embeddings
with open("data/embeddings.pkl", "rb") as f:
    embeddings_np, labels = pickle.load(f)
# Load image paths
with open("data/images_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

# Builds a cosine similarity index using
faiss.normalize_L2(embeddings_np)

index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

# Transformation for input image
transform = val_transforms

@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Get embedding
        with torch.no_grad():
            embedding = model(image_tensor)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            query_np = embedding.cpu().numpy().astype(np.float32)

        # Search
        similarity_scores, indices = index.search(query_np, 5)
        results = [
            {
                "image_path": image_paths[int(i)],
                "similarity_score": float(round(score, 5))
            }
            for i, score in zip(indices[0], similarity_scores[0])
        ]

        # Sort descending by similarity_score (though Faiss returns it that way)
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# # Entry point
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)