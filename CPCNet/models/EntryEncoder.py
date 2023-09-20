import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPool2D


class EntryEncoderCL(Layer):
    """
    0707
    """

    def __init__(self, image_size, channels):
        super(EntryEncoderCL, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.output_size = [-1, 16, int(self.image_size / 8), int(self.image_size / 8), self.channels[1]]
        self.conv1 = Conv2D(channels[0], kernel_size = 7, strides = 2, padding = "same", use_bias = False, data_format = "channels_last", input_shape = [self.image_size, self.image_size, 1])
        self.bn1 = BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2D(pool_size = 3, strides = 2, padding = "same", data_format = "channels_last")

        self.conv2 = Conv2D(channels[1], kernel_size = 3, strides = 1, padding = "same", use_bias = False,  data_format = "channels_last")
        self.bn2 = BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2D(pool_size = 3, strides = 2, padding = "same", data_format = "channels_last")

    def call(self, x, training):

        x = tf.reshape(x, [-1, self.image_size, self.image_size, 1])  # need to use this because max pooling does not support extended batch shape.
        # x = tf.expand_dims(x, axis = -1)

        x = self.conv1(x)
        x = self.bn1(x, training = training)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = tf.reshape(x, self.output_size)
        return x
