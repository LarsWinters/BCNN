# This is a sample Python script.

#here are all imports necessary for the CNN
import tensorflow as tf
import os
import glob
from PIL import Image
import pandas as pd

#define paths --> data should be in github repo --> need to be changed
train_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'
test_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'
pred_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'

def train_folders():
    for folder in os.listdir(train_path + 'seg_train'):
        files = glob.glob(pathname=train_path + "seg_train//" + folder + '/*.jpg')
        #print(f'({folder}) folder has: {len(files)}')
    return

def train_files():
    train_img_size = []
    for folder in os.listdir(train_path + 'seg_train'):
        files = glob.glob(pathname = train_path + "seg_train//" + folder + '/*.jpg')
        for file in files:
            im = Image.open(file)
            train_img_size.append(im.size)
    pd.Series(train_img_size).value_counts()
    return
#model architecture

def main():
    train_folders()
    train_files()

if __name__ == "__main__":
    main()