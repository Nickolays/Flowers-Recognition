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