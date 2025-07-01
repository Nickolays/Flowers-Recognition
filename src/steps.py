import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import 


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        orig = batch['anchor'].to(device)
        pos = batch['positive'].to(device)
        negs = batch['negatives'].to(device)  # [B, N, C, H, W]
        B, N, C, H, W = negs.shape

        # Flatten negatives for encoding
        negs = negs.view(B * N, C, H, W)

        # Forward pass
        anchor_emb = model(orig)              # [B, D]
        pos_emb = model(pos)                  # [B, D]
        negs_emb = model(negs).view(B, N, -1) # [B, N, D]

        loss = contrastive_loss_dot(anchor_emb, pos_emb, negs_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            orig = batch['anchor'].to(device)
            pos = batch['positive'].to(device)
            negs = batch['negatives'].to(device)
            B, N, C, H, W = negs.shape
            negs = negs.view(B * N, C, H, W)

            anchor_emb = model(orig)
            pos_emb = model(pos)
            negs_emb = model(negs).view(B, N, -1)

            loss = contrastive_loss_dot(anchor_emb, pos_emb, negs_emb)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def inference_embeddings(model, dataloader, device):
    """Extract normalized embeddings and paths."""
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding"):
            orig = batch['anchor'].to(device)
            emb = model(orig)
            emb = torch.nn.functional.normalize(emb, dim=1)  # cosine similarity support
            embeddings.append(emb.cpu())
            labels.append(batch['class_idx'])
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)
    return embeddings, labels