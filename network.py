from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, ELU, BatchNormalization, Dropout, AveragePooling2D
from tensorflow.keras import initializers, regularizers
import tensorflow as tf

class model(tf.keras.Model):
    def __init__(self, classes, batch_size, seed=123, l2_weight=1e-4):
        super(model, self).__init__()

        weights_initializer = initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='uniform', seed=seed)
        weights_regularizer = regularizers.L1L2

        # conv1-1
        self.conv1_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', input_shape=[32, 32, 1],
                                kernel_initializer=weights_initializer,
                                kernel_regularizer=weights_regularizer(l1=0.0, l2=l2_weight))
        self.B1_1 = BatchNormalization(momentum=0.99)
        self.E1_1 = ELU()
        # conv1-2
        self.conv1_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='valid',
                                kernel_initializer=weights_initializer,
                                kernel_regularizer=weights_regularizer(l1=0.0, l2=l2_weight))
        self.B1_2 = BatchNormalization(momentum=0.99)
        self.E1_2 = ELU()
        self.P1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # conv2-1
        self.conv2_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                                kernel_initializer=weights_initializer,
                                kernel_regularizer=weights_regularizer(l1=0.0, l2=l2_weight))
        self.B2_1 = BatchNormalization(momentum=0.99)
        self.E2_1 = ELU()
        # conv2-2
        self.conv2_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
                                kernel_initializer=weights_initializer,
                                kernel_regularizer=weights_regularizer(l1=0.0, l2=l2_weight))
        self.B2_2 = BatchNormalization(momentum=0.99)
        self.E2_2 = ELU()
        self.P2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # conv3-1
        self.conv3_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='valid',
                                kernel_initializer=weights_initializer,
                                kernel_regularizer=weights_regularizer(l1=0.0, l2=l2_weight))
        self.B3_1 = BatchNormalization(momentum=0.99)
        self.E3_1 = ELU()
        # drop out
        self.dropout = Dropout(rate=0.1, noise_shape=(batch_size, 1, 1, 128), seed=None)
        self.conv3_2 = Conv2D(128, (3, 3), strides=(1, 1), padding='valid',
                                kernel_initializer=weights_initializer,
                                kernel_regularizer=weights_regularizer(l1=0.0, l2=l2_weight))
        self.B3_2 = BatchNormalization(momentum=0.99)
        self.E3_2 = ELU()

        self.AP = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))
        self.flat = Flatten(name='final_feat')

        # clf
        self.fc1 = Dense(64)
        self.E4 = ELU()
        self.fc2 = Dense(classes, activation='softmax')

    def encoder(self, x):
        # conv1-1
        x = self.E1_1(self.B1_1(self.conv1_1(x)))
        # conv1-2
        x = self.P1(self.E1_2(self.B1_2(self.conv1_2(x))))
        # conv2-1
        x = self.E2_1(self.B2_1(self.conv2_1(x)))
        # conv2-2
        x = self.P2(self.E2_2(self.B2_2(self.conv2_2(x))))
        # conv3-1
        x = self.E3_1(self.B3_1(self.conv3_1(x)))
        # drop out
        x = self.dropout(x)
        # conv3-2
        x = self.flat(self.AP(self.E3_2(self.B3_2(self.conv3_2(x)))))
        return x

    def cls(self, x):
        x = self.fc1(x)
        x = self.E4(x)
        x = self.fc2(x)
        return x

    def call(self, inputs):
        feat = self.encoder(inputs)
        pred = self.cls(feat)
        return feat, pred