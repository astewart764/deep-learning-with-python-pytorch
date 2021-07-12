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

class EstateDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])


class RegressionNetwork(nn.Module):
    def __init__(self, in_feats = 13, out_feats = 1):
        super(RegressionNetwork, self).__init__()

        self.input      =   nn.Linear(in_features = in_feats, out_features = 20)
        self.fc1        =   nn.Linear(in_features = 20, out_features = 30)
        self.fc2        =   nn.Linear(in_features = 30, out_features = 15)
        self.out        =   nn.Linear(in_features = 15, out_features = out_feats)

        self.relu       =   nn.ReLU()
        self.sigmoid    =   nn.Sigmoid()

    def forward(self, x):
        x   =   x.float()
        x   =   self.input(x)
        x   =   self.relu(x)
        x   =   self.fc1(x)
        x   =   self.relu(x)
        x   =   self.fc2(x)
        x   =   self.relu(x)
        x   =   self.out(x)
        x   =   self.relu(x)

        return x


def normalise_data(data):
    data_features, data_labels = data
    mu = data_features.mean(axis = 0)
    sigma = data_features.std(axis = 0)
    new_data_features = (data_features - mu) / sigma

    return (new_data_features, data_labels)


def k_splitting(data, k_splits = 4):
    data_features, data_labels = data
    fold_size = math.floor(len(data_features) / k_splits)
    folded_data = []
    folded_labels = []
    for k in range(k_splits):
        if len(folded_data) == 0:
            folded_data, folded_labels = data_features[k * fold_size : (k+1) * fold_size], data_labels[k * fold_size : (k+1) * fold_size]
            folded_data, folded_labels = folded_data[:, :, np.newaxis], folded_labels[:, np.newaxis]

        else:
            folded_data, folded_labels = np.insert(folded_data, k, data_features[k * fold_size : (k+1) * fold_size], axis = 2), np.insert(folded_labels, k, data_labels[k * fold_size : (k+1) * fold_size], axis = 1)

    return folded_data, folded_labels


def get_device():
    device_name = CUDA.get_device_name()
    if CUDA.is_available():
        device = 'cuda:0'
        print("CUDA device available : Using " + device_name + "\n")

    else:
        device = 'cpu'
        print("CUDA device unavailable : Using " + device_name + "\n")

    return device, device_name


def smooth_points(data, theta = 0.4):
    smoothed_points = []

    for point in data:
        if len(smoothed_points) > 1:
            previous = smoothed_points[-1]
            new_point = (previous * theta) + (point * (1 - theta))
            smoothed_points.append(new_point)

        else:
            smoothed_points.append(point)

    smoothed_points = smoothed_points[2:]
    return smoothed_points


def training_procedure(model, data, device, epochs, kwargs):
    criterion   =   nn.MSELoss()
    optimizer   =   Adam(model.parameters(), lr = 0.0005)

    Path(str(Path(__file__).parent.parent) + '\\.models\\').mkdir(parents = True, exist_ok = True)

    previous_loss = np.finfo(np.float32).max

    data_features, data_labels = data

    training_history = list()
    validation_history = []

    iter_counter    =   0
    for epoch in range(epochs):
        running_loss    =   0

        for k in range(data_features.shape[2]):
            val_features, val_labels = data_features[:, :, k], data_labels[:, k]
            lower_flag = 0
            upper_flag = 0
            if k > 0:
                lower_set, lower_labels = data_features[:, :, :k], data_labels[:, :k]
                lower_set = np.reshape(lower_set, (lower_set.shape[0] * lower_set.shape[2], lower_set.shape[1]))
                lower_labels = np.reshape(lower_labels, (lower_labels.shape[0] * lower_labels.shape[1]))
                lower_flag = 1
            if k+1 < data_features.shape[2]:
                upper_set, upper_labels = data_features[:, :, k+1:], data_labels[:, k+1:]
                upper_set = np.reshape(upper_set, (upper_set.shape[0] * upper_set.shape[2], upper_set.shape[1]))
                upper_labels = np.reshape(upper_labels, (upper_labels.shape[0] * upper_labels.shape[1]))
                upper_flag = 1

            if lower_flag == 1:
                if upper_flag == 1:
                    training_features = torch.from_numpy(np.concatenate((lower_set, upper_set), axis = 0))
                    training_labels = torch.from_numpy(np.concatenate((lower_labels, upper_labels), axis = 0))
                else:
                    training_features = lower_set
                    training_labels = lower_labels
            else:
                training_features = upper_set
                training_labels = upper_labels

            trainset    =   EstateDataset(training_features, training_labels)
            trainloader =   DataLoader(trainset, **kwargs)

            for batch_n, (inputs, labels) in enumerate(trainloader):
                optimizer.zero_grad()

                outputs         =   model(inputs).reshape(kwargs['batch_size'])
                labels          =   labels.to(dtype = torch.float)
                loss            =   criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss    +=  loss.item()

                if loss < previous_loss:
                    print('\nSaving Model...\n')
                    torch.save(model.state_dict(), str(Path(__file__).parent.parent) + '\\.models\\real_estate_checkpoint.pt')
                    previous_loss = loss
                    save_point = iter_counter

            print("Epoch: {:2d} | Fold: {:3d} | Batch: {:2d} | Loss: {:3.3f}".format(epoch + 1, k+1, batch_n+1, running_loss))
            running_loss    =   0
            iter_counter    +=  1

            with torch.no_grad():
                training_array = model(training_features).reshape(len(training_features))
                training_accuracy = (training_array - training_labels) / training_labels
                training_accuracy = abs(float(torch.sum(training_accuracy) / len(training_features)))
                training_history.append(training_accuracy)

                validation_array = model(val_features).reshape(len(val_features))
                validation_accuracy = (validation_array - val_labels) / val_labels
                validation_accuracy = abs(float(torch.sum(validation_accuracy) / len(val_features)))
                validation_history.append(validation_accuracy)

    return(training_history, validation_history, save_point)


def main():

    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    train_tuple = (train_data, train_labels)
    train_norm = normalise_data(train_tuple)

    train_split_norm = k_splitting(train_norm, k_splits = 4)
    train_data, train_labels = train_split_norm

    DEVICE, device_name = get_device()

    x_train = torch.from_numpy(train_data)
    y_train = torch.from_numpy(train_labels)

    training_inputs = (x_train, y_train)

    network = RegressionNetwork(in_feats = 13, out_feats = 1)

    kwargs = {'batch_size'  : 4,
              'shuffle'     : True,
              'num_workers' : 0,
              'drop_last'   : True,
              'pin_memory'  : False,
             }

    training_history, validation_history, save_point = training_procedure(network, training_inputs, DEVICE, 40, kwargs)
    #training_history = smooth_points(training_history)
    #validation_history = smooth_points(validation_history)

    print("Plotting Training History...")
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(training_history)), training_history, color = 'black', linestyle = '-', label = 'Training Error')
    ax.plot(range(len(validation_history)), validation_history, color = 'grey', linestyle = '-', label = 'Validation Error')
    ax.axvline(save_point, color = 'green', linestyle = '--', label = 'Save Point')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
