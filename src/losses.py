import torch
import torch.nn.functional as F

def contrastive_loss_dot(anchor, positive, negatives, temperature=0.07, eps=1e-8):
    """
    Improved contrastive loss with ViT and full normalization stability.
    anchor:   [B, D]
    positive: [B, D]
    negatives: [B, N, D]
    """
    B, D = anchor.size()
    N = negatives.size(1)

    # Normalize all inputs (avoid NaNs by clamping norm)
    anchor = F.normalize(anchor + eps, dim=1)
    positive = F.normalize(positive + eps, dim=1)
    negatives = F.normalize(negatives + eps, dim=2)  # [B, N, D]

    # Positive logits: [B, 1]
    pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)

    # Negative logits: [B, N]
    neg_logits = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)

    # Scale logits
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    logits = logits / temperature

    # Numerical stability: subtract max for softmax safety
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Targets: class 0 is positive
    labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

    # Compute cross-entropy
    loss = F.cross_entropy(logits, labels)
    return loss

def contrastive_loss_nt_xent(anchor, positive, negatives, temperature=0.07, eps=1e-8):
    """
    Contrastive loss (NT-Xent) для batch с anchor, positive и negatives.

    anchor:   [B, D]
    positive: [B, D]
    negatives: [B, N, D]

    Возвращает: скалярный loss
    """
    B, D = anchor.size()
    N = negatives.size(1)

    # Нормализация эмбеддингов
    anchor = F.normalize(anchor + eps, dim=1)      # [B, D]
    positive = F.normalize(positive + eps, dim=1)  # [B, D]
    negatives = F.normalize(negatives + eps, dim=2)  # [B, N, D]

    # Положительные логиты: косинусное сходство anchor-positive
    pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)  # [B, 1]

    # Отрицательные логиты: косинусное сходство anchor-negatives
    neg_logits = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # [B, N]

    # Объединяем положительные и отрицательные логиты
    logits = torch.cat([pos_logits, neg_logits], dim=1)  # [B, 1+N]

    # Масштабируем по температуре
    logits = logits / temperature

    # Метки: положительный класс — 0
    labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

    # Вычисляем кросс-энтропию
    loss = F.cross_entropy(logits, labels)
    return loss


# def contrastive_loss_nt_xent(anchor, positive, negatives=None, temperature=0.07):
#     """
#     NT-Xent loss (InfoNCE) for contrastive learning.
#     anchor:   [B, D]
#     positive: [B, D]
#     negatives: [B, N, D] or None (negatives included in batch)
#     """
#     anchor = F.normalize(anchor, dim=1)
#     positive = F.normalize(positive, dim=1)

#     # Compute logits
#     pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)  # [B,1]

#     if negatives is not None:
#         neg = F.normalize(negatives, dim=2)  # [B,N,D]
#         neg_logits = torch.bmm(anchor.unsqueeze(1), neg.permute(0, 2, 1)).squeeze(1)
#         logits = torch.cat([pos_logits, neg_logits], dim=1)
#     else:
#         logits = pos_logits  # Self-supervised without negatives

#     logits = logits / temperature
#     labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)

#     return F.cross_entropy(logits, labels)