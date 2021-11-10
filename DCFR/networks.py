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
                 small_hidden=256, 
                 large_hidden=512,
                 learning_rate=0.0001):
        self.__small_hidden = small_hidden
        self.__large_hidden = large_hidden
        self.__init_weights = initializers.RandomUniform(minval=-0.005, maxval=0.005)
        self.__path_to_save = './models/'
        self.__activation = activation

        self.LEARNING_RATE = learning_rate
        self.HIST_INPUT = env.BET_HISTORY_LENGTH 
        self.BATCH_SIZE = 500
        self.EPOCHS = 16_000

        self.__env = env
        self.model = self.create_network()

    def create_network(self):
        input_layer = Input(shape=(len(self.__env.CARDS), 3))
        flat = Flatten()(input_layer)
        dense_first = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(flat)
        dense_second = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                   activation='relu')(dense_first)

        dense_last = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                 activation='relu')(dense_second)
        norm = BatchNormalization()(dense_last)
        output = Dense(self.__env.ACTIONS_NUM, activation=self.__activation)(norm)

        model = Model(inputs=input_layer,outputs=output)

        def loss_func(y_true, y_pred):
            output, timestep = y_true[0], y_true[1]
            return timestep*(output - y_pred)**2

        model.compile(loss=loss_func,
                      optimizer=Adam(learning_rate=self.LEARNING_RATE, clipnorm=1))
        return model

    def predict(self, hole, board, hist):
        data = np.stack((hole, board, hist), axis=-1)
        return self.model.predict(data[np.newaxis, :])

    def train_net(self, info_set, output, tensorboard=None):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)

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
