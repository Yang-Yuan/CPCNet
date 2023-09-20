import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU, Dropout, Flatten, AveragePooling2D

from .standard_modules import ResBlockCL
from .CenterContrast import SymmetricContrast1D_MLP
from .EntryEncoder import EntryEncoderCL

RCHW_perm_CF = [0, 4, 5, 3, 1, 2]
RCHW_perm_CL = [0, 3, 4, 1, 2, 5]


class CPCNet(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_2 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_3 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_4 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_5a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_5a = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.conv_hw_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)
        self.conv_rc_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.contrast_5 = SymmetricContrast1D_MLP(self.channels)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])
        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_1(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_2(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_3a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_3a(x_a, training = training)

        x_b = self.conv_hw_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_3(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_4a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_4a(x_a, training = training)

        x_b = self.conv_hw_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_4(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_5a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_5a(x_a, training = training)

        x_b = self.conv_hw_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_5(x_a, x_b)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_b = self.mlp_b(x_b)
        x_a = tf.reshape(x_a, shape = [-1, 8])
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_a, x_b


class CPCNet_0_Contrasting_Layer(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_0_Contrasting_Layer, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])
        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x = tf.reduce_mean(x, axis = -1)
        x = tf.reshape(x, [-1, 900])

        x_a = self.mlp_a(x)
        x_b = self.mlp_b(x)
        x_a = tf.reshape(x_a, shape = [-1, 8])
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_a, x_b


class CPCNet_1_Contrasting_Layer(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_1_Contrasting_Layer, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])
        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_1(x_a, x_b)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_b = self.mlp_b(x_b)
        x_a = tf.reshape(x_a, shape = [-1, 8])
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_a, x_b


class CPCNet_2_Contrasting_Layer(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_2_Contrasting_Layer, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_2 = SymmetricContrast1D_MLP(self.channels)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])
        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_1(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_2(x_a, x_b)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_b = self.mlp_b(x_b)
        x_a = tf.reshape(x_a, shape = [-1, 8])
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_a, x_b


class CPCNet_3_Contrasting_Layer(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_3_Contrasting_Layer, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_2 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_3 = SymmetricContrast1D_MLP(self.channels)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])
        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_1(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_2(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_3a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_3a(x_a, training = training)

        x_b = self.conv_hw_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_3(x_a, x_b)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_b = self.mlp_b(x_b)
        x_a = tf.reshape(x_a, shape = [-1, 8])
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_a, x_b


class CPCNet_4_Contrasting_Layer(Model):
        """
        Channel-last implementation.
        """

        def __init__(self, image_size, channels):
            super(CPCNet_4_Contrasting_Layer, self).__init__()

            self.image_size = image_size
            self.channels = channels

            self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

            self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

            self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.contrast_2 = SymmetricContrast1D_MLP(self.channels)

            self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.contrast_3 = SymmetricContrast1D_MLP(self.channels)

            self.conv_rc_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_hw_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.conv_hw_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
            self.conv_rc_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

            self.contrast_4 = SymmetricContrast1D_MLP(self.channels)

            self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                     Dense(units = 1, activation = None, use_bias = True)])
            self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                     Dense(units = 1, activation = None, use_bias = True)])

        def call(self, x, training):
            """
            Forward pass.
            """

            x = self.entry_encoder(x, training = training)

            context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
            context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
            choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
            x = tf.concat([context, choices], axis = 2)

            x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

            x_a = tf.transpose(x, perm = RCHW_perm_CL)
            x_a = self.conv_rc_1a(x_a, training = training)
            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_hw_1a(x_a, training = training)

            x_b = self.conv_hw_1b(x, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
            x_b = self.conv_rc_1b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

            x_a, x_b = self.contrast_1(x_a, x_b)

            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_rc_2a(x_a, training = training)
            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_hw_2a(x_a, training = training)

            x_b = self.conv_hw_2b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
            x_b = self.conv_rc_2b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

            x_a, x_b = self.contrast_2(x_a, x_b)

            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_rc_3a(x_a, training = training)
            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_hw_3a(x_a, training = training)

            x_b = self.conv_hw_3b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
            x_b = self.conv_rc_3b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

            x_a, x_b = self.contrast_3(x_a, x_b)

            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_rc_4a(x_a, training = training)
            x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
            x_a = self.conv_hw_4a(x_a, training = training)

            x_b = self.conv_hw_4b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
            x_b = self.conv_rc_4b(x_b, training = training)
            x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

            x_a, x_b = self.contrast_4(x_a, x_b)

            x_a = tf.reduce_mean(x_a, axis = -1)
            x_a = tf.reshape(x_a, [-1, 900])

            x_b = tf.reduce_mean(x_b, axis = -1)
            x_b = tf.reshape(x_b, [-1, 900])

            x_a = self.mlp_a(x_a)
            x_b = self.mlp_b(x_b)
            x_a = tf.reshape(x_a, shape = [-1, 8])
            x_b = tf.reshape(x_b, shape = [-1, 8])

            return x_a, x_b


class CPCNet_LP(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_LP, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_5a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_5a = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_3a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_3a(x_a, training = training)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_4a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_4a(x_a, training = training)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_5a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_5a(x_a, training = training)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_a = tf.reshape(x_a, shape = [-1, 8])

        return x_a


class CPCNet_UP(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_UP, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)
        self.conv_rc_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_b = self.conv_hw_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_b = self.conv_hw_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_b = self.conv_hw_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_b = self.mlp_b(x_b)
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_b


class CPCNet_IC(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_IC, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_rc_5a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_5a = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.conv_hw_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)
        self.conv_rc_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])
        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_3a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_3a(x_a, training = training)

        x_b = self.conv_hw_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_4a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_4a(x_a, training = training)

        x_b = self.conv_hw_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_5a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_5a(x_a, training = training)

        x_b = self.conv_hw_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_b = self.mlp_b(x_b)
        x_a = tf.reshape(x_a, shape = [-1, 8])
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_a, x_b


class CPCNet_UC(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_UC, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_2 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_3 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_4 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_5a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_5a = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.conv_hw_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)
        self.conv_rc_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.contrast_5 = SymmetricContrast1D_MLP(self.channels)

        self.mlp_b = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_1(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_2(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_3a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_3a(x_a, training = training)

        x_b = self.conv_hw_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_3(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_4a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_4a(x_a, training = training)

        x_b = self.conv_hw_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_4(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_5a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_5a(x_a, training = training)

        x_b = self.conv_hw_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        _, x_b = self.contrast_5(x_a, x_b)

        x_b = tf.reduce_mean(x_b, axis = -1)
        x_b = tf.reshape(x_b, [-1, 900])

        x_b = self.mlp_b(x_b)
        x_b = tf.reshape(x_b, shape = [-1, 8])

        return x_b


class CPCNet_LC(Model):
    """
    Channel-last implementation.
    """

    def __init__(self, image_size, channels):
        super(CPCNet_LC, self).__init__()

        self.image_size = image_size
        self.channels = channels

        self.entry_encoder = EntryEncoderCL(image_size = self.image_size, channels = [int(self.channels / 2), self.channels])

        self.conv_rc_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_1a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_1b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_1 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_2a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_2b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_2 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_3a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_3b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_3 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_4a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.conv_hw_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_rc_4b = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)

        self.contrast_4 = SymmetricContrast1D_MLP(self.channels)

        self.conv_rc_5a = ResBlockCL(filter_n = self.channels, kernel_size = 3, stride = 1)
        self.conv_hw_5a = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.conv_hw_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)
        self.conv_rc_5b = ResBlockCL(filter_n = self.channels, kernel_size = 5, stride = 1)

        self.contrast_5 = SymmetricContrast1D_MLP(self.channels)

        self.mlp_a = Sequential([Dense(units = 128, activation = "relu", use_bias = True),
                                 Dense(units = 1, activation = None, use_bias = True)])

    def call(self, x, training):
        """
        Forward pass.
        """

        x = self.entry_encoder(x, training = training)

        context = tf.expand_dims(x[:, :8, :, :, :], axis = 1)
        context = tf.tile(context, multiples = [1, 8, 1, 1, 1, 1])
        choices = tf.expand_dims(x[:, 8:, :, :, :], axis = 2)
        x = tf.concat([context, choices], axis = 2)

        x = tf.reshape(x, shape = [-1, 3, 3, 10, 10, self.channels])

        x_a = tf.transpose(x, perm = RCHW_perm_CL)
        x_a = self.conv_rc_1a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_1a(x_a, training = training)

        x_b = self.conv_hw_1b(x, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_1b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_1(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_2a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_2a(x_a, training = training)

        x_b = self.conv_hw_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_2b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_2(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_3a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_3a(x_a, training = training)

        x_b = self.conv_hw_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_3b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_3(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_4a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_4a(x_a, training = training)

        x_b = self.conv_hw_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_4b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, x_b = self.contrast_4(x_a, x_b)

        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_rc_5a(x_a, training = training)
        x_a = tf.transpose(x_a, perm = RCHW_perm_CL)
        x_a = self.conv_hw_5a(x_a, training = training)

        x_b = self.conv_hw_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)
        x_b = self.conv_rc_5b(x_b, training = training)
        x_b = tf.transpose(x_b, perm = RCHW_perm_CL)

        x_a, _ = self.contrast_5(x_a, x_b)

        x_a = tf.reduce_mean(x_a, axis = -1)
        x_a = tf.reshape(x_a, [-1, 900])

        x_a = self.mlp_a(x_a)
        x_a = tf.reshape(x_a, shape = [-1, 8])

        return x_a
