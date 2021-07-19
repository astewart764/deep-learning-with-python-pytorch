from pathlib import Path
import os
import shutil
import random
from PIL import Image
import numpy as np

class ReviewDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        categories = {0 : 'Negative', 1 : 'Positive'}
        return (self.data[idx], self.labels[idx])

numpy_directory = str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats/numpy_datasets/"
train_set = np.load(numpy_directory + 'dogs-vs-cats-train.npy')
