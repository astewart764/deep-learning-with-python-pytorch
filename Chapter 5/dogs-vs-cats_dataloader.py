from pathlib import Path
import os
import shutil
import random
from PIL import Image
import numpy as np

def CreateSet(category = 'dog', count = 3000):
    set_directory = str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats/train"

    if category.startswith('dog'):
        data_directory = set_directory + '/dogs'
        label = 1

    elif category.startswith('cat'):
        data_directory = set_directory + '/cats'
        label = 0

    file_names = [data_directory + '/' + file for file in os.listdir(data_directory)[:count]]
    file_labels = [label for file in os.listdir(data_directory)[:count]]
    comb_list = list(zip(file_names, file_labels))
    random.shuffle(comb_list)
    file_names, file_labels = zip(*comb_list)

    return list(file_names), list(file_labels)

def CropSet(dataset_files, target_dim = 256):

    img_stack = []
    for idx, file in enumerate(dataset_files):
        img = np.asarray(Image.open(file))

        min_dim = min(img.shape[0:2])
        max_dim = max(img.shape[0:2])

        scaling = target_dim / min_dim
        new_size = (round(scaling * img.shape[1]), round(scaling * img.shape[0]))
        img = Image.fromarray(img).resize(new_size)

        width, height = new_size
        top = round((height / 2) - (target_dim / 2))
        bottom = round((height / 2) + (target_dim / 2))
        left = round((width / 2) - (target_dim / 2))
        right = round((width / 2) + (target_dim / 2))
        img = img.crop((left, top, right, bottom))
        img = np.asarray(img)
        img = img.reshape(3, target_dim, target_dim)

        if len(img_stack) == 0:
            img_stack = img
            img_stack = img_stack[np.newaxis, :, :, :]

        else:
            img_stack = np.insert(img_stack, idx, img, axis = 0)

    return img_stack

dataset_directory       =   str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats"
training_directory      =   dataset_directory + "/train"
testing_directory     =   dataset_directory + "/test"

dogs_training_directory =   training_directory + "/dogs"
cats_training_directory =   training_directory + "/cats"
dogs_testing_directory  =   testing_directory + "/dogs"
cats_testing_directory  =   testing_directory + "/cats"

Path(dogs_training_directory).mkdir(parents = True, exist_ok = True)
Path(cats_training_directory).mkdir(parents = True, exist_ok = True)
Path(dogs_testing_directory).mkdir(parents = True, exist_ok = True)
Path(cats_testing_directory).mkdir(parents = True, exist_ok = True)

dog_training_files = [file for file in os.listdir(training_directory) if file.startswith('dog.')]
cat_training_files = [file for file in os.listdir(training_directory) if file.startswith('cat.')]

for dog_file in dog_training_files:
    shutil.move(training_directory + '/' + dog_file, dogs_training_directory + '/' + dog_file)

for cat_file in cat_training_files:
    shutil.move(training_directory + '/' + cat_file, cats_training_directory + '/' + cat_file)

dog_set, dog_labels = CreateSet('dogs', count = 1500)
cat_set, cat_labels = CreateSet('cats', count = 1500)

dog_stack = CropSet(dog_set, 224)
cat_stack = CropSet(cat_set, 224)

full_stack = np.concatenate([dog_stack, cat_stack], axis = 0)
full_labels = np.concatenate([dog_labels, cat_labels], axis = 0)
stack_zip = list(zip(full_stack, full_labels))
random.shuffle(stack_zip)
full_stack, full_labels = zip(*stack_zip)
full_stack, full_labels = np.asarray(full_stack), np.asarray(full_labels)

train_stack = full_stack[:2000, :, :, :]
valid_stack = full_stack[2000:2500, :, :, :]
test_stack = full_stack[2500:, :, :, :]

train_labels = full_labels[:2000]
valid_labels = full_labels[2000:2500]
test_labels = full_labels[2500:]

numpy_directory = str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats/numpy_datasets/"

np.save(numpy_directory + 'dogs-vs-cats-train_set.npy', train_stack)
np.save(numpy_directory + 'dogs-vs-cats-valid_set.npy', valid_stack)
np.save(numpy_directory + 'dogs-vs-cats-test_set.npy', test_stack)

np.save(numpy_directory + 'dogs-vs-cats-train_labels.npy', train_labels)
np.save(numpy_directory + 'dogs-vs-cats-valid_labels.npy', valid_labels)
np.save(numpy_directory + 'dogs-vs-cats-test_labels.npy', test_labels)
