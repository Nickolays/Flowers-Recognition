import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import ViTContrastive, contrastive_loss_dot
from src.dataset import ContrastiveDataset
from src.steps import train_one_epoch, validate, inference_embeddings


# Main Training Runner
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ContrastiveDataset('data/flowers', n_neg=3)
val_dataset = ContrastiveDataset('data/flowers', n_neg=3)    # data/flowers/val

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

model = ViTContrastive(pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

best_loss = float('inf')
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, 11):
    print(f"\nEpoch {epoch}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f"checkpoints/best_model.pt")
        print("Saved best model!")

# Optional: inference after training
embeddings, labels = inference_embeddings(model, val_loader, device)
print("Embedding shape:", embeddings.shape)