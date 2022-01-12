# CNN submitted by Nicola Schreyer, Kathrin Heldauer, Jannik Holz, Niklas Grimm, Paul Baßler, Lucas Winkler
# here are all imports necessary for the CNN
import os
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import glob
from PIL import Image
import pandas as pd
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
#K.clear_session()
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Define Classes for Image Classification Model
classes = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

# define image size
img_size = 150

# define path
path = 'C:/Users/c4371094/Desktop/Lucas_CNN/archive/'

# list train folder structure
def train_folders():
    for folder in os.listdir(path + 'seg_train'):
        files = glob.glob(pathname=path + 'seg_train//' + folder + '/*.jpg')  # * = .glob module wildcard
        print(f'({folder}) folder has: {len(files)}')
    return


# list train image shapes
def train_files():
    train_img_size = []
    for folder in os.listdir(path + 'seg_train'):
        files = glob.glob(pathname=path + 'seg_train//' + folder + '/*.jpg')
        for file in files:
            im = Image.open(file)
            train_img_size.append(im.size)
    print('Train Image shapes:\n')
    train_series = pd.Series(train_img_size).value_counts()
    print(train_series, '\n')
    return


# list test folder structure
def test_folders():
    for folder in os.listdir(path + 'seg_test'):
        files = glob.glob(pathname=path + 'seg_test//' + folder + '/*.jpg')
        print(f'({folder}) folder has: {len(files)}')
    return


# list test image shapes
def test_files():
    test_img_size = []
    for folder in os.listdir(path + 'seg_test'):
        files = glob.glob(pathname=path + 'seg_test//' + folder + '/*.jpg')
        for file in files:
            im = Image.open(file)
            test_img_size.append(im.size)
    print('Test Image shapes:\n')
    test_series = pd.Series(test_img_size, name='Height x Width').value_counts()
    print(test_series, '\n')
    return


# list prediction folder structure
def pred_folders():
    files = glob.glob(pathname=path + 'seg_pred//' + '*.jpg')
    print(f'Prediction folder has: {len(files)}')
    return


# list pred image shapes
def pred_files():
    pred_img_size = []
    files = glob.glob(pathname=str(path + 'seg_pred/*.jpg'))
    for file in files:
        im = Image.open(file)
        pred_img_size.append(im.size)
    print('Prediction Image shapes:\n')
    pred_series = pd.Series(pred_img_size).value_counts()
    print(pred_series, '\n')
    return

# setup x_train, y_train
def setup_train():
    x_train = []
    y_train = []
    for folder in os.listdir(path + "seg_train"):
        files = glob.glob(pathname=path + "seg_train//" + folder + "/*.jpg")
        for file in files:
            img = cv2.imread(file)
            img_array = cv2.resize(img, (img_size, img_size))
            x_train.append(list(img_array))
            y_train.append(classes[folder])
    return x_train, y_train


# setup x_test, y_test
def setup_test():
    x_test = []
    y_test = []
    for folder in os.listdir(path + "seg_test"):
        files = glob.glob(pathname=path + "seg_test//" + folder + "/*.jpg")
        for file in files:
            img = cv2.imread(file)
            img_array = cv2.resize(img, (img_size, img_size))
            x_test.append(list(img_array))
            y_test.append(classes[folder])
    return x_test, y_test


# setup x_pred
def setup_pred():
    x_pred = []
    files = glob.glob(pathname=path + 'seg_pred//' + '*.jpg')
    for file in files:
        img = cv2.imread(file)
        img_array = cv2.resize(img, (img_size, img_size))
        x_pred.append(list(img_array))
    return x_pred


# print array shape of prepared data
def data_array_shape(x_train, y_train, x_test, y_test, x_pred):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_pred = np.array(x_pred)

    print(f'x_train array shape is {x_train.shape}')
    print(f'y_train array shape is {y_train.shape}')
    print(f'x_test array shape is {x_test.shape}')
    print(f'y_test array shape is {y_test.shape}')
    print(f'x_pred array shape is {x_pred.shape}')
    return x_train, y_train, x_test, y_test, x_pred


# convolutional neural network architecture
def cnn_architecture():
    # network architecture
    model = Sequential()
    # input shape of images
    set_input_shape = (img_size, img_size, 3)
    #set_dropout = 0.3
    # model settings c1
    set_filters_c1 = 128
    set_kernel_c1 = (5, 5)
    set_actfunc_c1 = 'elu'
    set_poolsize_c1 = (2, 2)
    # model settings c2
    set_filters_c2 = 256
    set_kernel_c2 = (3, 3)
    set_actfunc_c2 = 'elu'
    set_poolsize_c2 = (2, 2)
    # model settings prediction block
    set_units_d1 = 64
    set_units_d3 = 32
    set_units_d4 = 16
    set_units_d5 = 6
    set_actfunc_d1 = 'elu'
    set_actfunc_d3 = 'elu'
    set_actfunc_d4 = 'elu'
    set_actfunc_d5 = 'softmax'

    # model layers
    # conv_block c1
    model.add(Conv2D(set_filters_c1, kernel_size=set_kernel_c1,
                     activation=set_actfunc_c1,
                     input_shape=set_input_shape))
    model.add(MaxPooling2D(pool_size=set_poolsize_c1))
    model.add(BatchNormalization())
    # conv_block c2
    model.add(Conv2D(set_filters_c2, kernel_size=set_kernel_c2,
                     activation=set_actfunc_c2))
    model.add(MaxPooling2D(set_poolsize_c2))
    # model.add(Dropout(set_dropout)
    model.add(BatchNormalization())

    # flatten
    model.add(Flatten())
    model.add(Dense(set_units_d1, set_actfunc_d1, name='features'))
    model.add(BatchNormalization())
    model.add(Dense(set_units_d3, set_actfunc_d3))
    model.add(Dense(set_units_d4, set_actfunc_d4))
    model.add(Dense(set_units_d5, set_actfunc_d5))
    return model


# compile model with loss function and optimization
def model_compilation(model):
    # sparse categorical cross entropy because of integer classes [0,1,2,3,4,5]
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    print('Model compilation successful')
    return model


def create_tensorboard():
    logdir = "logs/scalars/"
    my_tensorboard = TensorBoard(log_dir=logdir,
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True)
    print('Tensorboard creation sucessful')
    return my_tensorboard


def model_training(model, my_tensorboard, x_train, y_train, x_test, y_test):
    # hyperparameters
    set_batch_size = 6  # only divisor of 14034 (training sample size) without remainders
    set_epochs = 10
    model_history = model.fit(x_train, y_train,
                              batch_size = set_batch_size,
                              callbacks=[my_tensorboard],
                              epochs=set_epochs,
                              verbose=1,
                              validation_data=(x_test, y_test))
    pd.DataFrame(model_history.history).plot()
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    return model_history

def model_evaluation(model_history):
    score = model_history.evaluate(x_test, y_test)
    print('Test Loss: ', score[0])
    print('Test Accuracy ', score[1])
    return


def main():
    print(tf.__version__)
    tf.debugging.set_log_device_placement(True) # shows operations of used device while running
    """
    print('Train Dataset:\n')
    train_folders()
    print()
    train_files()
    print('Test Dataset:\n')
    test_folders()
    print()
    test_files()
    print('Prediction Dataset:\n')
    pred_folders()
    print()
    pred_files()
    """
    global x_train, y_train, x_test, y_test, x_pred
    print('----------------------Setup x_train, y_train, x_test, y_test, x_pred'
          '--------------------------------------\n')
    try:
        x_train, y_train = setup_train()
        print('Successfully created x_train, y_train!')
    except:
        print('Setup x_train, y_train failed!')
    try:
        x_test, y_test = setup_test()
        print('Successfully created x_test, y_test!')
    except:
        print('Setup x_test, y_test failed!')
    try:
        x_pred = setup_pred()
        print('Successfully created x_pred!')
    except:
        print('Setup x_pred failed!')
    x_train, y_train, x_test, y_test, x_pred = data_array_shape(x_train, y_train, x_test, y_test, x_pred)
    model = cnn_architecture()
    model.summary()
    model_compilation(model)
    my_tensorboard = create_tensorboard()
    model_history = model_training(model, my_tensorboard, x_train, y_train, x_test, y_test)
    model_evaluation(model_history)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Execution Time:\n')
    print("--- %s seconds ---" % (time.time() - start_time))
