from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


class Poker_network:

    def __init__(self, 
                 num_actions,
                 cards_len,
                 activation,
                 small_hidden=256, 
                 large_hidden=512,
                 learning_rate=0.0001):
        self.__small_hidden = small_hidden
        self.__large_hidden = large_hidden
        self.__init_weights = initializers.RandomUniform(minval=-0.01, maxval=0.01)
        self.__path_to_save = './models/'
        self.__activation = activation
        self.__learning_rate = learning_rate

        self.__batch_size = 500
        self.__epochs = 5_000

        self.__num_actions = num_actions
        self.__cards_len = cards_len
        self.model = self.__create_network()

    @staticmethod
    def loss_func(y_true, y_pred):
        output, timestep = y_true[:, :-1], y_true[:, -1]
        value =  tf.math.reduce_mean(
                tf.math.multiply(
                    ((output - y_pred)**2),
                    timestep[:, np.newaxis])
        )
        return value

    def __create_network(self):
        input_layer = Input(shape=(self.__cards_len, 3))
        flat = Flatten()(input_layer)
        dense_first = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(flat)
        dense_second = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                   activation='relu')(dense_first)

        dense_last = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                 activation='relu')(dense_second)
        norm = BatchNormalization()(dense_last)
        output = Dense(self.__num_actions, activation=self.__activation)(norm)

        model = Model(inputs=input_layer,outputs=output)


        model.compile(loss=Poker_network.loss_func,
                      optimizer=Adam(learning_rate=self.__learning_rate, clipnorm=1),
                      run_eagerly=True)
        return model

    def predict(self, hole, board, hist):
        data = np.stack((hole, board, hist), axis=-1)
        return self.model.predict(data[np.newaxis, :])

    def train_net(self, info_set, output, timesteps, tensorboard=None):
        early_stop = EarlyStopping(monitor='val_loss', patience=40)

        if tensorboard:
            callbacks = [early_stop, tensorboard]
        else:
            callbacks = [early_stop]
        
        timesteps = np.array(timesteps)[:, np.newaxis]
        out_merge = np.append(output, timesteps, axis=1)
        with tf.device('/gpu:0'):
            self.model.fit(np.array(info_set), 
                           out_merge,
                           validation_split=0.2,
                           shuffle=True,
                           batch_size=self.__batch_size,
                           epochs=self.__epochs,
                           verbose=1,
                           callbacks=callbacks)

    def save_model(self, name):
        self.model.save(self.__path_to_save+name)

    def clear_net(self):
        self.model = self.__create_network()
