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

def get_device():
    device_name = CUDA.get_device_name()
    if CUDA.is_available():
        device = 'cuda:0'
        print("CUDA device available : Using " + device_name + "\n")

    else:
        device = 'cpu'
        print("CUDA device unavailable : Using " + device_name + "\n")

    return device, device_name

def training_procedure(model, data, device, epochs, kwargs):
    criterion   =   nn.BCELoss()
    optimizer   =   Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

    Path(str(Path(__file__).parent.parent) + '\\.models\\').mkdir(parents = True, exist_ok = True)

    previous_loss = np.finfo(np.float32).max

    for epoch in range(epochs):
        running_loss    =   0
        iter_counter    =   0

        trainset    =   ReviewDataset(*data)
        trainloader =   DataLoader(trainset, **kwargs)

        for inputs, labels in trainloader:
            optimizer.zero_grad()

            outputs         =   model(inputs).reshape(kwargs['batch_size'])
            labels          =   labels.to(dtype = torch.float)
            loss            =   criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss    +=  loss.item()
            iter_counter    +=  1

            if iter_counter % 4 == 3:
                accuracy        =   [int(outputs[idx] == labels[idx]) for idx in range(kwargs['batch_size'])]
                accuracy        =   int(sum(accuracy)/len(accuracy) * 100)
                print("Epoch: {:3d} | Ieration : {:5d} | Loss : {:3.3f} | Accuracy : {}".format(epoch + 1, iter_counter + 1, running_loss, accuracy))
                running_loss    =   0

            if loss < previous_loss:
                print('\nSaving Model...\n')
                torch.save(model.state_dict(), str(Path(__file__).parent.parent) + '\\.models\\IMDb_checkpoint.pt')
                previous_loss = loss


def vectorise_sequences(sequences, dimension = 20000, d_type = np.int64):
    vector = np.zeros((len(sequences), dimension), dtype = d_type)

    for (idx, sequence) in enumerate(sequences):
        vector[idx, sequence] = 1

    return vector

def main():

    DEVICE, device_name = get_device()

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 20000)

    '''
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join(reverse_word_index.get(i - 3, '?') for i in train_data[0])
    '''

    x_train = torch.from_numpy(vectorise_sequences(train_data, 20000, np.int64)).to(DEVICE)
    y_train = torch.from_numpy(np.asarray(train_labels, dtype = np.int64)).to(DEVICE)

    train_data = (x_train, y_train)

    network = FFNetwork(in_feats = 20000, out_feats = 1).to(DEVICE)

    kwargs = {'batch_size'  : 16,
              'shuffle'     : True,
              'num_workers' : 0,
              'drop_last'   : True,
              'pin_memory'  : False,
             }

    training_procedure(network, train_data, DEVICE, 1, kwargs)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
