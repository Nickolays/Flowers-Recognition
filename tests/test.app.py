import torch
import numpy as np
from src.model import ViTContrastive
from src.steps import inference_embeddings
from src.dataset import ContrastiveDataset
from torch.utils.data import DataLoader
import faiss

torch.manual_seed(777)

def test_model_forward():
    model = ViTContrastive()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 128)
    assert not torch.isnan(out).any()

def test_inference_embeddings():
    dummy_dataset = ContrastiveDataset("flowers/test", n_neg=0)
    dummy_loader = DataLoader(dummy_dataset, batch_size=2)
    model = ViTContrastive().eval()
    embeddings, labels = inference_embeddings(model, dummy_loader, device="cpu")
    assert embeddings.shape[0] == len(dummy_dataset)
    assert embeddings.shape[1] == 128
    assert labels.shape[0] == len(dummy_dataset)

def test_faiss_index():
    data = np.random.rand(10, 128).astype(np.float32)
    faiss.normalize_L2(data)
    index = faiss.IndexFlatIP(128)
    index.add(data)
    query = data[:2]
    scores, indices = index.search(query, 3)
    assert indices.shape == (2, 3)
    assert scores.shape == (2, 3)