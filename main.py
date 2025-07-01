import torch, yaml, os, pickle
from torch.utils.data import DataLoader
import faiss
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.model import ViTContrastive
from src.dataset import ContrastiveDataset
from src.steps import train_one_epoch, validate, inference_embeddings


# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ViTContrastive()
model.load_state_dict(torch.load(cfg['model_path'], map_location=device))
model.to(device)
model.eval()

# Load validation dataset
val_dataset = ContrastiveDataset(cfg['val_data_dir'], n_neg=3)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Get embeddings
embeddings, labels = inference_embeddings(model, val_loader, device)  # shape [N, D]

# Build Faiss index
embeddings_np = embeddings.numpy().astype(np.float32)
faiss.normalize_L2(embeddings_np)  # required for cosine similarity
index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

# Visualize top-5 results for 5 random query images
transform = val_dataset.transform
samples = [val_dataset[i] for i in torch.randint(0, len(val_dataset), (5,))]

for i, sample in enumerate(samples):
    query_tensor = sample['original'].unsqueeze(0).to(device)
    query_emb = model(query_tensor)
    query_emb = torch.nn.functional.normalize(query_emb, dim=1)
    query_np = query_emb.cpu().numpy().astype(np.float32)

    _, top5_indices = index.search(query_np, 5)   # Return distance and indices

    # Plot results
    plt.figure(figsize=(15, 3))
    plt.subplot(1, 6, 1)
    plt.imshow(transforms.ToPILImage()(sample['original']))
    plt.title("Query")
    plt.axis('off')

    for j, idx in enumerate(top5_indices[0]):
        img_path, _ = val_dataset.dataset.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        plt.subplot(1, 6, j + 2)
        plt.imshow(img)
        plt.title(f"Top {j+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()