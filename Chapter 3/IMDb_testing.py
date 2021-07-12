import torch
import torch.nn as nn
import torch.cuda as CUDA
import torch.nn.functional as F
import numpy as np
from keras.datasets import imdb
from torch.optim import RMSprop, Adam
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from pathlib import Path

class ReviewDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        categories = {0 : 'Negative', 1 : 'Positive'}
        return (self.data[idx], self.labels[idx])

class FFNetwork(nn.Module):
    def __init__(self, in_feats = 20000, out_feats = 1):
        super(FFNetwork, self).__init__()

        self.input      =   nn.Linear(in_features = in_feats, out_features = 16)
        self.fc1        =   nn.Linear(in_features = 16, out_features = 16)
        self.out        =   nn.Linear(in_features = 16, out_features = out_feats)

        self.relu       =   nn.ReLU()
        self.sigmoid    =   nn.Sigmoid()

    def forward(self, x):
        x   =   x.float()
        x   =   self.input(x)
        x   =   self.relu(x)
        x   =   self.fc1(x)
        x   =   self.relu(x)
        x   =   self.out(x)
        x   =   self.sigmoid(x)

        return x

def vectorise_sequences(sequences, dimension = 20000, d_type = np.int64):
    vector = np.zeros((len(sequences), dimension), dtype = d_type)

    for (idx, sequence) in enumerate(sequences):
        vector[idx, sequence] = 1

    return vector

def get_device():
    device_name = CUDA.get_device_name()
    if CUDA.is_available():
        device = 'cuda:0'
        print("CUDA device available : Using " + device_name + "\n")

    else:
        device = 'cpu'
        print("CUDA device unavailable : Using " + device_name + "\n")

    return device, device_name

def testing_procedure(network, data, device):

    x_data, y_data = data

    inputs  =   torch.tensor(x_data, dtype = torch.float32)

    with torch.no_grad():
        outputs  =   network(x_data)
        predictions  =   torch.round(outputs).numpy().astype(int)

    return predictions, y_data

def main(sample_size = 100):

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 20000)

    x_test = torch.from_numpy(vectorise_sequences(test_data, 20000, np.int64))
    y_test = torch.from_numpy(np.asarray(test_labels, dtype = np.int64))

    test_data = (x_test[:sample_size], y_test[:sample_size])

    network = FFNetwork(in_feats = 20000, out_feats = 1)
    network.load_state_dict(torch.load(str(Path(__file__).parent.parent) + '\\.models\\IMDb_checkpoint.pt'))

    predictions, labels = testing_procedure(network, test_data, 'cpu')
    predictions = predictions.squeeze()
    accuracy        =   [int(predictions[idx] == labels[idx]) for idx in range(len(labels))]
    accuracy        =   sum(accuracy)/len(accuracy) * 100
    print("Accuracy of Model : {:3.1f}%".format(accuracy))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main(5000)
