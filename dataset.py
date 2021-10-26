import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms

class rnn_custom_dataset(Dataset):
    
    def __init__(self, data) -> None:
        super().__init__()
        xy = data
        self.x = xy[:, :, :-1].astype("float32")
        self.y = xy[:, :, -1].astype("float32")
        self.n_samples = xy.shape[0]
        del xy
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples








