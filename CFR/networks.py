from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
import sys

sys.path.append('../environment')
from environment.emulator import params

'''
Neural Network with 3 inputs (hole cards, board cards, history of bets),
hole cards and board cards are passed as one hot encodet array with size 52,
'''


class Poker_network:

    def __init__(self, hist_bet_length = 20,
                 small_hidden=64, 
                 large_hidden=192,
                 learning_rate=0.0001):
        self.__small_hidden = small_hidden
        self.__large_hidden = large_hidden
        self.__init_weights = initializers.Zeros()

        self.LEARNING_RATE = learning_rate
        self.HIST_INPUT = hist_bet_length # one hot of raise = 1 nad call = 0

        self.model = self.create_network()

    def create_network(self):
        hole_input_layer = Input(shape=params.NUM_HOLE_CARDS)
        board_input_layer = Input(shape=params.NUM_BOARD_CARDS) 

        hole_dense = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                           activation='relu')(hole_input_layer)
        board_dense = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                            activation='relu')(board_input_layer)

        conc_cards_layer = Concatenate()([hole_dense, board_dense])
        cards_dense_first = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(conc_cards_layer)
        cards_dense_second = Dense(self.__large_hidden, kernel_initializer=self.__init_weights,
                                   activation='relu')(cards_dense_first)
        cards_dense_third = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(cards_dense_second)

        hist_input_layer = Input(shape=self.HIST_INPUT)
        dense_hist_first = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                 activation='relu')(hist_input_layer)
        dense_hist_second = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                  activation='relu')(dense_hist_first)

        conc_hist_cards = Concatenate()([dense_hist_second, cards_dense_third])

        dense_all_first = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                activation='relu')(conc_hist_cards)
        dense_all_second = Dense(self.__small_hidden, kernel_initializer=self.__init_weights,
                                 activation='relu')(dense_all_first)
        output = Dense(params.ACTIONS_NUM, activation='linear')(dense_all_second)

        model = Model(inputs=[hole_input_layer, board_input_layer, hist_input_layer],
                      outputs=output)

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.LEARNING_RATE))
        return model

    def predict(self, hole, board, hist):
        return self.model.predict([hole, board, hist])
