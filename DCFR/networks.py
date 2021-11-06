from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

class Poker_network:

    def __init__(self, 
                 env,
                 activation,
                 small_hidden=128, 
                 large_hidden=256,
                 learning_rate=0.001):
        self.__small_hidden = small_hidden
        self.__large_hidden = large_hidden
        self.__init_weights = initializers.Zeros()
        self.__path_to_save = './models/'
        self.__activation = activation

        self.LEARNING_RATE = learning_rate
        self.HIST_INPUT = env.BET_HISTORY_LENGTH 
        self.BATCH_SIZE = 6_000
        self.EPOCHS = 32_000

        self.__env = env
        self.model = self.create_network()

    def create_network(self):
        input_layer = Input(shape=(len(self.__env.CARDS), 3))
        flat = Flatten()(input_layer)
        dense_first = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(flat)
        dense_second = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                   activation='relu')(dense_first)
        dense_third = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                   activation='relu')(dense_second)
        dense_fourth = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(dense_third)

        dense_last = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                 activation='relu')(dense_fourth)
        norm = BatchNormalization()(dense_last)
        output = Dense(self.__env.ACTIONS_NUM, activation=self.__activation)(norm)

        model = Model(inputs=input_layer,outputs=output)

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.LEARNING_RATE))
        return model

    def predict(self, hole, board, hist):
        data = np.stack((hole, board, hist), axis=-1)
        return self.model.predict(data[np.newaxis, :])

    def train_net(self, info_set, output, tensorboard=None):
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        if tensorboard:
            callbacks = [early_stop, tensorboard]
        else:
            callbacks = [early_stop]
            
        with tf.device('/gpu:0'):
            self.model.fit(np.array(info_set), 
                           np.array(output),
                           validation_split=0.2,
                           shuffle=True,
                           batch_size=self.BATCH_SIZE,
                           epochs=self.EPOCHS,
                           verbose=1,
                           callbacks=callbacks)

    def save_model(self, name):
        self.model.save(self.__path_to_save+name)

    def clear_net(self):
        self.model = self.create_network()
