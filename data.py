import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    class MyDataset(Dataset):
        def __init__(self, filepaths):
            self.imgs = np.concatenate([np.load(f)["images"] for f in filepaths])
            self.labels = np.concatenate([np.load(f)["labels"] for f in filepaths])
        def __len__(self):
            return self.imgs.shape[0]

        def __getitem__(self, idx):
            return (self.imgs[idx], self.labels[idx])

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    
    train_files = ["train_0.npz", "train_1.npz", "train_2.npz", "train_3.npz", "train_4.npz"]
    train_files = [os.path.join("corruptmnist",f) for f in train_files]
    test_file = ["corruptmnist/test.npz"]

    train_dl = DataLoader(MyDataset(train_files), batch_size = 16, transform=transform)
    test_dl = DataLoader(MyDataset(test_file), batch_size = 16, transform=transform)
    return train_dl, test_dl
