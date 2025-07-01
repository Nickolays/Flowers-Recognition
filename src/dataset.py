from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import random, torch


# Base transforms (resize, normalize, etc.)
base_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class ContrastiveDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_neg=1):
        """
        root_dir: path to folder structure as per ImageFolder
        transform: augmentation transform
        n_neg: number of negative samples per anchor
        """
        self.dataset = ImageFolder(root=root_dir)
        self.transform = transform or base_transforms
        self.n_neg = n_neg

        # Build a map: class_idx -> list of indices
        self.class_to_indices = {}
        for idx, (_, class_idx) in enumerate(self.dataset.imgs):
            self.class_to_indices.setdefault(class_idx, []).append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Anchor
        path, class_idx = self.dataset.imgs[idx]
        img = Image.open(path).convert('RGB')
        orig = self.transform(img)

        # Positive: sample from same class
        pos_candidates = self.class_to_indices[class_idx].copy()
        pos_candidates.remove(idx)
        pos_idx = random.choice(pos_candidates)
        pos_img = Image.open(self.dataset.imgs[pos_idx][0]).convert('RGB')
        pos = self.transform(pos_img)

        # Negatives: sample images from other classes
        negs = []
        while len(negs) < self.n_neg:
            neg_class = random.choice([c for c in self.classes if c != class_idx])
            neg_idx = random.choice(self.class_to_indices[neg_class])
            neg_img = Image.open(self.dataset.imgs[neg_idx][0]).convert('RGB')
            negs.append(self.transform(neg_img))

        # Stack negatives into tensor [n_neg, C, H, W]
        negs = torch.stack(negs, dim=0)

        return {
            "anchor": orig,
            "positive": pos,
            "negatives": negs,
            "class_idx": class_idx
        }
