# This is a sample Python script.

#here are all imports necessary for the CNN
import tensorflow as tf

#model architecture
def main():
    hello = tf.constant("hello, Tensorflow!")
    tf.print(hello)
    print(hello)
    print("Tensorflow Version: {}".format(tf.__version__))

if __name__ == "__main__":
    main()