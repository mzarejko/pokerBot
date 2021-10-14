from collections import deque
from random import shuffle

class Memory:

    def __init__(self, size=4_000_000):
        self.__storage = deque(maxlen=size)

    def sample_batch(self, batch_size=10_000):
        return shuffle(list(self.__storage[:batch_size]))

    def append(self, info_set, regrets):
        self.__storage.append(info_set, regrets)


        
        

        
        
        

