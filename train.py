import torch, yaml, os, pickle
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics import RetrievalMAP, RetrievalPrecision

from src.model import ViTContrastive
from src.dataset import ContrastiveDataset
from src.steps import train_one_epoch, validate, inference_embeddings
from src.metrics import precision_at_k, recall_at_k, mean_average_precision, ndcg_at_k

torch.manual_seed(777)

# Load configuration
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Main Training Runner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = ContrastiveDataset(cfg['train_dir'], n_neg=cfg['n_negative'])
val_dataset = ContrastiveDataset(cfg['valid_dir'], n_neg=cfg['n_negative'])    # data/flowers/val

train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

model = ViTContrastive(pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Инициализация метрик
map_metric = RetrievalMAP()
precision_k_metric = RetrievalPrecision(top_k=5)  # precision@5

best_loss = float('inf')
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

    # Compute metrics
    val_embeddings, val_labels = inference_embeddings(model, val_loader, device)
    val_embeddings_np = val_embeddings.numpy()
    val_labels_np = val_labels.numpy()

    similarity_matrix = val_embeddings_np @ val_embeddings_np.T
    np.fill_diagonal(similarity_matrix, -1)   # исключаем совпадения с самим собой

    preds_list = []
    target_list = []
    indexes_list = []

    # Формируем данные для метрик
    for i in range(len(val_labels)):
        true_label = val_labels[i].item()
        scores = similarity_matrix[i]
        # Считаем релевантность: 1, если метка совпадает, иначе 0
        relevance = (val_labels.numpy() == true_label).astype(int)

        preds_list.extend(scores.tolist())
        target_list.extend(relevance.tolist())
        indexes_list.extend([i] * len(scores))  # все элементы принадлежат запросу i

    # Преобразуем в тензоры
    preds_tensor = torch.tensor(preds_list, dtype=torch.float32)
    target_tensor = torch.tensor(target_list, dtype=torch.bool)
    indexes_tensor = torch.tensor(indexes_list, dtype=torch.long)

    # Вычисляем метрики
    map_score = map_metric(preds_tensor, target_tensor, indexes=indexes_tensor)
    precision_k_score = precision_k_metric(preds_tensor, target_tensor, indexes=indexes_tensor)

    print(f"Validation mAP: {map_score.item():.4f}")
    print(f"Validation Precision@5: {precision_k_score.item():.4f}")

    # Сброс метрик для следующей эпохи
    map_metric.reset()
    precision_k_metric.reset()
    scheduler.step()

# Optional: inference after training
# embeddings, labels = inference_embeddings(model, val_loader, device)
# print("Embedding shape:", embeddings.shape)    # [4317, 128])

# # Save as nunpy in picke format
# with open("data/embeddings.pkl", "wb") as f:
#     pickle.dump(( embeddings.numpy().astype(np.float32), labels), f)