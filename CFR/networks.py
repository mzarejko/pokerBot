from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import sys
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.callbacks import History, EarlyStopping

sys.path.append('../environment')
from environment.emulator import params

'''
Neural Network with 3 inputs (hole cards, board cards, history of bets),
hole cards and board cards are passed as one hot encodet array with size 52,
'''


class Poker_network:

    def __init__(self, 
                 activation,
                 small_hidden=64, 
                 large_hidden=192,
                 learning_rate=0.0001):
        self.__small_hidden = small_hidden
        self.__large_hidden = large_hidden
        self.__init_weights = initializers.Zeros()
        self.__path_to_save = './models/'
        self.__activation = activation

        self.LEARNING_RATE = learning_rate
        self.HIST_INPUT = params.BET_HISTORY_LENGTH 
        self.BATCH_SIZE = 64
        self.EPOCHS = 500
        self.MIN_SIZE_TO_TRAIN = 5_000

        self.model = self.create_network()

    def create_network(self):
        input_layer = Input(shape=(len(params.CARDS), 3))
        flat = Flatten()(input_layer)
        dense_first = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(flat)
        dense_second = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                   activation='relu')(dense_first)
        dense_third = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(dense_second)

        dense_fourth = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                 activation='relu')(dense_third)
        norm_layer = BatchNormalization()(dense_fourth)
        output = Dense(params.ACTIONS_NUM, activation=self.__activation)(norm_layer)

        model = Model(inputs=input_layer,outputs=output)

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.LEARNING_RATE))
        return model

    def predict(self, hole, board, hist):
        data = np.stack((hole, board, hist), axis=-1)
        return self.model.predict(data[np.newaxis, :])

    def train_net(self, info_set, output, tensorboard=None):
        early_stop = EarlyStopping(monitor='val_loss', patience=2)

        if tensorboard:
            callbacks = [early_stop, tensorboard]
        else:
            callbacks = [early_stop]
            
        print('now train')
        with tf.device('/gpu:0'):
            self.model.fit(np.array(info_set), 
                           np.array(output),
                           validation_split=0.2,
                           shuffle=True,
                           batch_size=self.BATCH_SIZE,
                           epochs=self.EPOCHS,
                           verbose=1,
                           callbacks=callbacks)

    def update_advantage_tensorboard(self, value):
        self.mod_tensorboard.update_stats(advantage_loss = value)
        
    def save_model(self, name):
        self.model.save(self.__path_to_save+name)

    def clear_net(self):
        self.model = self.create_network()
