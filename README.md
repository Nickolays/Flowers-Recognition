# Flowers Recognition


## System Design
<img width=600 height=500 src="app_logic.png" alt="">

### What inside

    - ViT
    - Faiss(KNN)
    - Docker
    - FastAPI
    - pytest

## Example output
<img width=600 src="example.png" alt="">

supervised contrastive loss

## Code Examples

*Weights have to be here: models/best_model.pt*

### 1.Start application
```bash
python3 docker_utils.py
```

### 2. Inference application
```python
import requests

input_image_name = 'tests/test_img.jpg'
api_host = 'http://0.0.0.0:8000/'
type_rq = 'search'

files = {'file': open(input_image_name, 'rb')}

response = requests.post(api_host+type_rq, files=files)

data = response.json()     
print(data)
```

## Project structure
```
├── .gitignore
├── Dockerfile
├── README.md
├── app_logic.png
├── config.yaml
├── example.png
├── main.py
├── notebooks
    └── project.ipynb
├── requirements.txt
├── src
    ├── dataset.py
    ├── losses.py
    ├── metrics.py
    ├── model.py
    ├── steps.py
    └── utils.py
├── tests
    ├── test.app.py
    ├── test_img.jpeg
    └── test_main.py
└── train.py
```

## Что можно улучшить:

- Можно обучить модель побольше
- Quantizatiom, Prunning, Distilation
- Перенести вычисления Faiss на GPU.
- Если нам нужны более точные совпадения и мы не обрабатываем много  то можно использовать KNN
- Индексация повышает затраты памяти, потому что в индексной таблице сохраняются эмбеддинги целых изображений. Сократить затраты памяти позволяют различные методы оптимизации — такие, как векторное квантование и квантование по произведению (PQ, Product Quantization)
- Python плохо работает для быстрой обработки изображений, нет потоков и нормальнной ассинхронности, по-этому я бы заменил FastAPI на TorchServer.