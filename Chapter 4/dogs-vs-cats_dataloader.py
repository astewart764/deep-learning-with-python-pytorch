from pathlib import Path
import os
import shutil

dataset_directory       =   str(Path(__file__).parent.parent.parent) + "/data/dogs-vs-cats"
training_directory      =   dataset_directory + "/train"
testing_directoring     =   dataset_directory + "/test"

dogs_training_directory =   training_directory + "/dogs"
cats_training_directory =   training_directory + "/cats"
dogs_testing_directory  =   testing_directoring + "/dogs"
cats_testing_directory  =   testing_directoring + "/cats"

Path(dogs_training_directory).mkdir(parents = True, exist_ok = True)
Path(cats_training_directory).mkdir(parents = True, exist_ok = True)
Path(dogs_testing_directory).mkdir(parents = True, exist_ok = True)
Path(cats_testing_directory).mkdir(parents = True, exist_ok = True)

dog_training_files = [file for file in os.listdir(training_directory) if file.startswith('dog')]
cat_training_files = [file for file in os.listdir(training_directory) if file.startswith('cat')]

for dog_file in dog_training_files:
    print(training_directory + '/' + dog_file)
    print(dogs_training_directory + '/' + dog_file)
    shutil.move(training_directory + '/' + dog_file, dogs_training_directory + '/' + dog_file)
    input('..')

for cat_file in cat_training_files:
    shutil.move(training_directory + '/' + cat_file, cats_testing_directory + '/' + cat_file)

print(dog_training_files)
