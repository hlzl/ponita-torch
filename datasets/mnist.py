import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        self.mnist_dataset = datasets.MNIST(root='/home/hlzl/Data', train=train, transform=transform,
                                            target_transform=target_transform, download=download)

        # Precompute edge index
        h, w = 28, 28  # MNIST image size
        self.edge_index = torch.zeros((2, h * w * 4 - 4 * h), dtype=torch.long)
        count = 0
        for i in range(h):
            for j in range(w):
                if i > 0:
                    self.edge_index[:, count] = torch.tensor([i * w + j, (i - 1) * w + j])
                    count += 1
                if i < h - 1:
                    self.edge_index[:, count] = torch.tensor([i * w + j, (i + 1) * w + j])
                    count += 1
                if j > 0:
                    self.edge_index[:, count] = torch.tensor([i * w + j, i * w + j - 1])
                    count += 1
                if j < w - 1:
                    self.edge_index[:, count] = torch.tensor([i * w + j, i * w + j + 1])
                    count += 1

        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        self.pos = torch.stack([(x_grid - w // 2) / (w // 2), (y_grid - h // 2) / (h // 2)],
                                dim=-1).reshape(h*w, -1)

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]

        return {'pos': self.pos, 'x': transforms.functional.to_tensor(image).flatten(),
                'y': label, 'batch': idx, 'edge_index': self.edge_index}

# Usage
# train_loader = DataLoader(MNIST(train=True), batch_size=32, shuffle=True)
# test_loader = DataLoader(MNIST(train=False), batch_size=32, shuffle=False)