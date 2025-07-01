import torch
import torch.nn.functional as F


def contrastive_loss_dot(anchor, positive, negatives, temperature=0.07):
    """
    anchor:   [B, D]
    positive: [B, D]
    negatives: [B, N, D]
    """
    B, D = anchor.shape
    N = negatives.size(1)

    # Normalize all vectors
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=2)  # [B, N, D]

    # Positive logits: dot product between anchor and positive
    pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)  # [B, 1]

    # Negative logits: dot product between anchor and each negative
    neg_logits = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2)  # [B, N]

    # Combine and scale by temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature  # [B, 1+N]

    # Targets: 0 index is positive
    labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss