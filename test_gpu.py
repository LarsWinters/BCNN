import tensorflow as tf

def get_GPU_CPU_details():
    print("GPU vorhanden? ", tf.config.list_physical_devices('GPU'))
    print("Devices: ", tf.config.experimental.list_physical_devices())
    return

def test_gpu():
    hello = tf.constant('Hello, Tensorflow')
    tf.print(hello)
    print(hello)
    print('Tensorflow Version: {}'.format(tf.__version__))
    get_GPU_CPU_details()
    with tf.device('/device:GPU:0'):
        c = tf.constant([[0.0, 1.0, 2],[3,0,1]])
        d = tf.constant([[1.0,2.0],[4,6],[1,2]])
        res = tf.matmul(c,d)
        print(res)
    return

def output_num_gpu():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    return