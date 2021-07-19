from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, SubsetRandomSampler
from sys import getsizeof
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
import multiprocessing as mp
from torchvision.models import resnet34

def test(network, images, device):

    images  =   torch.tensor(images, dtype = torch.float32)

    with torch.no_grad():
        outputs  =   network(images)

    return outputs

def main():

    numpy_directory = str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats/numpy_datasets/"
    test_set = torch.from_numpy(np.load(numpy_directory + 'dogs-vs-cats-test_set.npy'))
    test_labels = torch.from_numpy(np.load(numpy_directory + 'dogs-vs-cats-test_labels.npy'))

    DEVICE  =  'cpu'

    pretrained_net = resnet34(pretrained = False)
    pretrained_net.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    pretrained_net.load_state_dict(torch.load(str(Path(__file__).parent.parent) + '\\.models\\cats-vs-dogs_ResNet_101.pt', map_location = DEVICE))

    predictions =   test(pretrained_net, test_set, DEVICE)
    predictions =   torch.reshape(predictions, (-1,))

    accuracy    =   (torch.sum(torch.round(predictions) == test_labels) / len(predictions)).cpu().numpy()
    print(round(accuracy * 100, 2))

    '''
    accuracy = 0
    for idx, lab in enumerate(labels):
        if lab == predictions[idx]:
            accuracy += 1
    accuracy = accuracy / len(predictions)

    fig1, axes  =   plt.subplots(nrows = 5, ncols = 4, gridspec_kw = {'hspace' : 0.5, 'wspace' : 0})
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.imshow(test_batch[idx, 0], cmap = 'gray')
        ax.set_title("Label : {} | Prediction : {}".format(labels[idx], predictions[idx]), fontsize = 8)
        ax.set_axis_off()
    fig1.suptitle('Accuracy : {:.0%}'.format(accuracy))

    plt.show()
    '''

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
