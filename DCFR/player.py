from .Memory import Memory
from .networks import Poker_network
from tensorflow.keras.callbacks import TensorBoard
import time

class Brain:

    def __init__(self, activation, env, uuid, memory_size=300_000):
        self.uuid = uuid
        self.__memory = Memory(size=memory_size)
        self.__net = Poker_network(env.ACTIONS_NUM, len(env.CARDS), activation) 

    def collect_samples(self, bet_history, hole_cards, community_cards, timestep, output):
        self.__memory.append(hole_cards, community_cards, bet_history, timestep, output)

    def train_net(self, verbose=False):
        self.__net.clear_net()
        info_set, timesteps, outputs = self.__memory.get_storage()
        if verbose:
            self.__net.train_net(info_set, outputs, timesteps, 
                                 tensorboard=TensorBoard(log_dir=f'./logs/{self.uuid}/{time.time()}'))
        else:
            self.__net.train_net(info_set, outputs, timesteps)

    def predict(self, hole, board, hist):
        return self.__net.predict(hole, board, hist).ravel()

    def save(self):
        self.__net.save_model(f'model_{self.uuid}_{time.time()}')
