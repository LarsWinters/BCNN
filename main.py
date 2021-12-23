# This is a sample Python script.

#here are all imports necessary for the CNN
import tensorflow as tf
import zipfile as zp
import os
import glob

#define paths --> data should be in github repo --> need to be changed
train_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'
test_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'
pred_path = 'C:/Users/Lucas/Desktop/Data_BIS_CNN/'

def train_folders():
    for folder in os.listdir(train_path + 'seg_train'):
        files = glob.glob(pathname=train_path + "seg_train//" + folder + '/*.jpg')
        print(f'({folder}) folder has: {len(files)}')
    return
#model architecture
def main():
    hello = tf.constant("hello, Tensorflow!")
    tf.print(hello)
    print(hello)
    print("Tensorflow Version: {}".format(tf.__version__))
    train_folders()

if __name__ == "__main__":
    main()