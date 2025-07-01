import torch, yaml, os, pickle
from torch.utils.data import DataLoader

from src.model import ViTContrastive
from src.dataset import ContrastiveDataset
from src.steps import train_one_epoch, validate, inference_embeddings


# Load configuration
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Main Training Runner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ContrastiveDataset(cfg['train_dir'], n_neg=cfg['negatives'])
val_dataset = ContrastiveDataset(cfg['train_dir'], n_neg=cfg['negatives'])    # data/flowers/val

train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

model = ViTContrastive(pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)

# best_loss = float('inf')
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, cfg['epochs'] + 1):
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
print("Embedding shape:", embeddings.shape)    # [4317, 128])

with open("data/embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)