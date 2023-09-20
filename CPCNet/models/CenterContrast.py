from tensorflow.keras.layers import Layer, BatchNormalization, Conv3D, Dense, Flatten


class SymmetricContrast1D_MLP(Layer):

    def __init__(self, units):
        super(SymmetricContrast1D_MLP, self).__init__()
        self.dense1 = Dense(units = units, activation = "relu", use_bias = True)
        self.dense2 = Dense(units = units, activation = None, use_bias = True)

    def call(self, x_a, x_b):
        y_a = self.dense2(self.dense1(x_a))
        y_b = self.dense2(self.dense1(x_b))
        x_a = x_a - y_b
        x_b = x_b - y_a
        return x_a, x_b

