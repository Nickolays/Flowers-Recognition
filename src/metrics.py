import numpy as np
import torch
# from torchmetrics.functional.retrieval import (
#     retrieval_precision,
#     retrieval_recall,
#     retrieval_map,
#     retrieval_ndcg
# )

# def compute_retrieval_metrics(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 5):
#     """
#     embeddings: [N, D] – нормализованные векторы
#     labels:     [N] – целочисленные метки классов
#     """
#     N = embeddings.size(0)

#     # Считаем матрицу попарных сходств
#     sim = embeddings @ embeddings.T  # [N, N]
#     sim.fill_diagonal_(-1)  # исключаем идентичные (само совпадение)

#     # Получаем топ-k индексы и оценки
#     topk_scores, topk_indices = torch.topk(sim, k, dim=1)  # [N, k]

#     # Подготовим входы для torchmetrics: flatten по всем query и кандидатам
#     preds = topk_scores.flatten()  # [N*k]
#     # целевое совпадение: True/False
#     true = (labels.unsqueeze(1) == labels[topk_indices]).flatten()
#     # индексы запросов
#     query_idx = torch.arange(N, device=embeddings.device).unsqueeze(1).repeat(1, k).flatten()

#     # Используем pytorch-metrics
#     prec = retrieval_precision(preds, true, top_k=k)
#     rec = retrieval_recall(preds, true, top_k=k)
#     mAP = retrieval_map(preds, true.int(), indexes=query_idx, top_k=k)
#     nDCG = retrieval_ndcg(preds, true.int(), indexes=query_idx, top_k=k)

#     return {
#         'precision@k': prec.item(),
#         'recall@k': rec.item(),
#         'mAP@k': mAP.item(),
#         'nDCG@k': nDCG.item()
#     }
    
def precision_at_k(y_true, y_pred, k):
    """
    y_true: List[int] — ground-truth labels (можно один элемент)
    y_pred: List[int] — top-k предсказанных меток
    k: int — значение k
    """
    y_pred = y_pred[:k]
    true_positives = sum(1 for pred in y_pred if pred in y_true)
    return true_positives / k

def recall_at_k(y_true, y_pred, k):
    """
    y_true: List[int] — ground-truth labels (можно один элемент)
    y_pred: List[int] — top-k предсказанных меток
    k: int — значение k
    """
    y_pred = y_pred[:k]
    true_positives = sum(1 for pred in y_pred if pred in y_true)
    return true_positives / len(y_true) if y_true else 0.0

def mean_average_precision(y_true, y_scores, k):
    """
    y_true: List[int] — ground-truth метки
    y_scores: List[float] — предсказанные scores (logits или cosine similarity)
    k: int — количество верхних примеров
    """
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    correct = 0
    ap = 0.0
    for i, idx in enumerate(sorted_indices):
        if idx in y_true:
            correct += 1
            ap += correct / (i + 1)
    return ap / min(len(y_true), k) if y_true else 0.0

def ndcg_at_k(y_true, y_scores, k):
    """
    y_true: List[int] — ground-truth метки
    y_scores: List[float] — предсказанные scores
    k: int — количество верхних элементов
    """
    sorted_indices = np.argsort(y_scores)[::-1][:k]
    gains = [1 if idx in y_true else 0 for idx in sorted_indices]
    
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))  # log2(i+2) т.к. i с 0
    ideal_gains = sorted([1] * min(len(y_true), k) + [0] * (k - min(len(y_true), k)), reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    
    return dcg / idcg if idcg > 0 else 0.0
