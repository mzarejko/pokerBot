from .Memory import Memory
from .networks import Poker_network
from tensorflow.keras.callbacks import TensorBoard
import time

class Brain:

    def __init__(self, activation, env, uuid):
        self.uuid = uuid
        self.__memory = Memory()
        self.__net = Poker_network(env, activation) 

    def collect_samples(self, bet_history, hole_cards, community_cards, output):
        self.__memory.append(hole_cards, community_cards, bet_history, output)

    def train_net(self, verbose=False):
        if self.__memory.is_enough_samples():
            self.__net.clear_net()
            info_set, outputs = self.__memory.get_storage()
            if verbose:
                self.__net.train_net(info_set, outputs, 
                                                TensorBoard(log_dir=f'logs/adv/{self.uuid}/{time.time()}'))
            else:
                self.__net.train_net(info_set, outputs)

    def predict(self, hole, board, hist):
        return self.__net.predict(hole, board, hist).ravel()

    def save(self):
        self.__net.save_model(f'model_{self.uuid}_{time.time()}')
