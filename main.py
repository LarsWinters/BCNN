# CNN submitted by Nicola Schreyer, Kathrin Heldauer, Jannik Holz, Niklas Grimm, Paul Baßler, Lucas Winkler
# here are all imports necessary for the CNN
import tensorflow as tf
import os
import glob
from PIL import Image
import pandas as pd

# define paths --> data should be in github repo --> need to be changed
train_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'
test_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'
pred_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'


# list train folder structure
def train_folders():
    for folder in os.listdir(train_path + 'seg_train'):
        files = glob.glob(pathname=train_path + 'seg_train//' + folder + '/*.jpg')  # * = .glob module wildcard
        print(f'({folder}) folder has: {len(files)}')
    return


# list train image shapes
def train_files():
    train_img_size = []
    for folder in os.listdir(train_path + 'seg_train'):
        files = glob.glob(pathname=train_path + 'seg_train//' + folder + '/*.jpg')
        for file in files:
            im = Image.open(file)
            train_img_size.append(im.size)
    print('Train Image shapes:\n')
    print(pd.Series(train_img_size).value_counts())
    return


# list test folder structure
def test_folders():
    for folder in os.listdir(test_path + 'seg_test'):
        files = glob.glob(pathname=test_path + 'seg_test//' + folder + '/*.jpg')
        print(f'({folder}) folder has: {len(files)}')
    return


# list test image shapes
def test_files():
    test_img_size = []
    for folder in os.listdir(test_path + 'seg_test'):
        files = glob.glob(pathname=test_path + 'seg_test//' + folder + '/*.jpg')
        for file in files:
            im = Image.open(file)
            test_img_size.append(im.size)
    print('Test Image shapes:\n')
    print(pd.Series(test_img_size).value_counts())
    return


# list prediction folder structure
def pred_folders():
    files = glob.glob(pathname=pred_path + 'seg_pred//' + '*.jpg')
    print(f'Prediction folder has: {len(files)}')
    return


# list pred image shapes
def pred_files():
    pred_img_size = []
    files = glob.glob(pathname=str(pred_path + 'seg_pred/*.jpg'))
    for file in files:
        im = Image.open(file)
        pred_img_size.append(im.size)
    print('Prediction Image shapes:\n')
    print(pd.Series(pred_img_size).value_counts())
    return


# model architecture

def main():
    print('Train Dataset:\n')
    train_folders()
    print()
    train_files()
    print()
    print('Test Dataset:\n')
    test_folders()
    print()
    test_files()
    print()
    print('Prediction Dataset:\n')
    pred_folders()
    print()
    pred_files()


if __name__ == '__main__':
    main()
