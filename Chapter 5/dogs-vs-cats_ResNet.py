import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import resnet101, resnet34
import torch.cuda as CUDA
from pathlib import Path
import os
import shutil
import random
from PIL import Image
import numpy as np
import time
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CatsVsDogsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])

def datloader_generator(train_data, val_data, device, kwargs):
    train_set, train_labels = train_data
    train_set, train_labels = torch.from_numpy(train_set).to(dtype = torch.float), torch.from_numpy(train_labels).to(dtype = torch.float)

    train_dataset = CatsVsDogsDataset(train_set, train_labels)
    train_dataloader =   DataLoader(train_dataset, **kwargs)

    if len(val_data[0]) != 0:
        val_set, val_labels = val_data
        val_set, val_labels = torch.from_numpy(val_set).to(dtype = torch.float), torch.from_numpy(val_labels).to(dtype = torch.float)
        val_dataset = CatsVsDogsDataset(val_set, val_labels)
        val_dataloader =   DataLoader(val_dataset, **kwargs)

        return {'train' : train_dataloader, 'val' : val_dataloader}

    else:
        return train_dataloader

def save_model(model):
    Path(str(Path(__file__).parent.parent) + '\\.models\\').mkdir(parents = True, exist_ok = True)

def get_device():
    device_name = CUDA.get_device_name()
    if CUDA.is_available():
        device = 'cuda:0'
        print("CUDA device available : Using " + device_name + "\n")

    else:
        device = 'cpu'
        print("CUDA device unavailable : Using " + device_name + "\n")

    return device, device_name

def train_model_with_val(model, dataloader, criterion, optimiser, scheduler, epochs, device_choice):

    model = model.to(device_choice)
    Path(str(Path(__file__).parent.parent) + '\\.models\\').mkdir(parents = True, exist_ok = True)

    previous_loss   =   np.finfo(np.float32).max
    update_interval =   50

    training_history = list()
    validation_history = list()

    for epoch in range(epochs):

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()

            elif phase == 'val':
                model.eval()

            running_loss        =   0
            running_accuracy    =   0
            iter_count          =   0

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device_choice)
                labels = labels.to(device_choice)
                optimiser.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs =   torch.reshape(model(inputs), (-1,))
                    loss    =   criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                running_loss        +=  loss.item()
                running_accuracy    +=  (torch.sum(torch.round(outputs) == labels) / len(outputs)).cpu()

                if iter_count % update_interval == 0 and iter_count > 0:
                    print("Phase: {:6} | Epoch: {:3d} | Batch : {:5d} | Loss : {:7.2f} | Accuracy : {:3.1f}".format(phase, epoch + 1, iter_count, running_loss, (running_accuracy / update_interval) * 100))

                    if phase == 'train':
                        training_history.append((running_accuracy / update_interval) * 100)

                        if running_loss < previous_loss:
                            print('\nSaving Model...\n')
                            torch.save(model.state_dict(), str(Path(__file__).parent.parent) + '\\.models\\cats-vs-dogs_ResNet_101.pt')
                            previous_loss = running_loss

                    else:
                        validation_history.append(running_loss)

                    running_loss    =   0
                    running_accuracy =   0

                iter_count += 1

            if phase == 'train':
                scheduler.step()

    return training_history

def main():
    DL_kwargs = {'batch_size'  : 4,
              'shuffle'     : True,
              'num_workers' : 0,
              'drop_last'   : True,
              'pin_memory'  : False,
             }

    device, _ = get_device()

    numpy_directory = str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats/numpy_datasets/"
    train_set = np.load(numpy_directory + 'dogs-vs-cats-train_set.npy')
    val_set = np.load(numpy_directory + 'dogs-vs-cats-valid_set.npy')
    train_labels = np.load(numpy_directory + 'dogs-vs-cats-train_labels.npy')
    val_labels = np.load(numpy_directory + 'dogs-vs-cats-valid_labels.npy')
    cat_dog_dataloader = datloader_generator((train_set, train_labels), (val_set, val_labels), device, DL_kwargs)

    pretrained_net = resnet34(pretrained = True)
    num_feats = pretrained_net.fc.in_features

    for parameter in pretrained_net.parameters():
        parameter.requires_grad = False
    pretrained_net.layer4.requires_grad = True
    pretrained_net.layer3.requires_grad = True
    
    pretrained_net.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    criterion = nn.BCELoss()
    optimiser = optim.RMSprop(pretrained_net.parameters(), lr = 0.01)
    scheduler = lr_scheduler.StepLR(optimiser, step_size=8, gamma=0.1)
    epochs = 30

    train_history = train_model_with_val(pretrained_net, cat_dog_dataloader, criterion, optimiser, scheduler, epochs, device)

    print("\nPlotting Training History...")
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(train_history)), train_history, color = 'black', linestyle = '-', label = 'Training Error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
