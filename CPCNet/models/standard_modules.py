from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, ReLU, Conv2D, BatchNormalization


class ResBlockCL(Layer):
    """
    A two-layer residual block.
    y = relu(conv_bn_1(x))
    y = conv_bn_2(y)
    z = relu( downsample(x) + y )
    """

    def __init__(self, filter_n, stride = 1, kernel_size = 3, input_shape = None, downsample = None):
        super(ResBlockCL, self).__init__()

        if input_shape is not None:
            conv1 = Conv2D(filter_n, kernel_size = kernel_size, strides = stride, padding = "same", data_format = "channels_last", use_bias = False, input_shape = input_shape)
        else:
            conv1 = Conv2D(filter_n, kernel_size = kernel_size, strides = stride, padding = "same", data_format = "channels_last", use_bias = False)
        bn1 = BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)
        relu1 = ReLU()
        conv2 = Conv2D(filter_n, kernel_size = kernel_size, strides = 1, padding = "same", use_bias = False,  data_format = "channels_last")
        bn2 = BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)
        self.conv1_bn1_relu1_conv2_bn2 = Sequential([conv1, bn1, relu1, conv2, bn2])

        if downsample is not None:
            self.downsample = downsample
        else:
            if 1 == stride:
                self.downsample = None
            else:
                self.downsample = Sequential([Conv2D(filter_n, kernel_size = 1, strides = stride, padding = "same", data_format = "channels_last", use_bias = False),
                                              BatchNormalization(axis = -1, momentum = 0.9, epsilon = 1e-5)])

        self.relu2 = ReLU()

    def call(self, x, training):

        y = self.conv1_bn1_relu1_conv2_bn2(x, training)

        if self.downsample is not None:
            x = self.downsample(x, training)

        z = self.relu2(x + y)

        return z

