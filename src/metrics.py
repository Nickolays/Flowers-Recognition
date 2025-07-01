import numpy as np


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
