from collections import deque
import numpy as np

class Memory:

    def __init__(self, size, min_size=1_00):
        self.__info_sets = deque(maxlen=size)
        self.__timesteps = deque(maxlen=size)
        self.__outputs = deque(maxlen=size)
        self.__min_size = min_size
        
    def append(self, hole, community, bet, timestep, outputs):
        data = np.stack((hole, community, bet), axis=-1)
        self.__info_sets.append(data)
        self.__timesteps.append(timestep-2)
        self.__outputs.append(outputs)

    def get_storage(self):
        if len(self.__info_sets) > self.__min_size:
            return self.__info_sets, self.__timesteps, self.__outputs

    def is_enough_samples(self):
        return len(self.__info_sets) > self.__min_size
