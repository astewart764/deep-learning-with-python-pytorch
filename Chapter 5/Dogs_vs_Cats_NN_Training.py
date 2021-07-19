import torch
import torch.nn as nn
import torch.cuda as CUDA
import torch.nn.functional as F
import numpy as np
from keras.datasets import boston_housing
from torch.optim import RMSprop, Adam
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        categories = {0 : 'cat', 1 : 'dog'}
        return (self.data[idx], self.labels[idx])

class Cats_vs_Dogs_ConvNet(nn.Module):
    def __init__(self, in_channels):
        super(Cats_vs_Dogs_ConvNet, self).__init__()

        self.conv1  =   nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2  =   nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3  =   nn.Conv2d(32, 64, 3, 1, 1)

        self.norm1  =   nn.BatchNorm2d(16)
        self.norm2  =   nn.BatchNorm2d(32)
        self.norm3  =   nn.BatchNorm2d(64)

        self.pool   =   nn.MaxPool2d(2, 2, 0)

        self.relu   =   nn.ReLU()

        self.fc1    =   nn.Linear(4800, 256)
        self.fc2    =   nn.Linear(256, 8)
        self.out    =   nn.Linear(8, 1)

        self.sig    =   nn.Sigmoid()

    def forward(self, x):

        x   =   self.conv1(x)
        x   =   self.relu(x)
        x   =   self.batch1(x)
        x   =   self.pool(x)

        x   =   self.conv2(x)
        x   =   self.relu(x)
        x   =   self.batch2(x)
        x   =   self.pool(x)

        x   =   self.conv3(x)
        x   =   self.relu(x)
        x   =   self.batch3(x)
        x   =   self.pool(x)

        x   =   x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])

        x   =   self.fc1(x)
        x   =   self.relu(x)

        x   =   self.fc2(x)
        x   =   self.relu(x)

        x   =   self.out(x)
        x   =   self.sig(x)

        return x

datasets_directory = str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats/numpy_datasets/"
training_set = np.load(datasets_directory + 'dogs-vs-cats-train_set.npy')
training_labels = np.load(datasets_directory + 'dogs-vs-cats-train_labels.npy')

training_dataset = ImageDataset(training_set, training_labels)
