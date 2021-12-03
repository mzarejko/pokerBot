from collections import deque
import numpy as np

class Memory:

    def __init__(self, size):
        self.__info_sets = deque(maxlen=size)
        self.__timesteps = deque(maxlen=size)
        self.__outputs = deque(maxlen=size)
        
    def append(self, hole, community, bet, timestep, outputs):
        data = np.stack((hole, community, bet), axis=-1)
        self.__info_sets.append(data)
        self.__timesteps.append(timestep-2)
        self.__outputs.append(outputs)

    def get_storage(self):
        return self.__info_sets, self.__timesteps, self.__outputs
