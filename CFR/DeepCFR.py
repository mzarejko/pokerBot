from enum import Enum, auto
from .networks import Poker_network
from .Memory import Memory
import numpy as np
import sys
from enum import Enum

sys.path.append('../environment')
from environment.emulator import Poker_limit_Emulator, params


class Players(Enum, str):
    PLAYER = 'player'
    OPPONENT = 'opponent'


class DCFR:

    def __init__(self, iterations, k):
        self.BET_HISTORY_LENGTH = 20
        self.bet_history = np.zeros([self.BET_HISTORY_LENGTH])
        
        self.advantage_memory = Memory()
        self.strategy_memory = Memory()
        self.advantage_net_player = Poker_network(self.BET_HISTORY_LENGTH)
        self.advantage_net_opponent = Poker_network(self.BET_HISTORY_LENGTH)
        self.strategy_net = Poker_network(self.BET_HISTORY_LENGTH)
        
        self.ITERATIONS = iterations
        self.K = k
        self.env = Poker_limit_Emulator(Players.PLAYER, Players.OPPONENT)

    def append_bet_to_history(self, timestemp):
        self.bet_history[timestemp] = 1

    def train(self):
        for t in self.ITERATIONS:
            state = self.env.initial_game()
            state, player_cards, opponent_cards = self.env.choose_cards(state)
            for p in params.NUM_PLAYERS:
                for k in self.K:
                    self.traverse(state, timestemp=0)
    

    def traverse(self, 
                 history,  
                 state, 
                 timestemp,
                 player_cards,
                 opponent_cards,
                 prob_player=1, 
                 prob_opponent=1, 
                 chance_prob=1):
        if self.env.is_terminal():
            return self.env.get_reward()

        if self.env.get_turn() == Players.PLAYER:
            action_utils = np.zeros([len(params.ACTIONS)])


            _, strategy = self.__predict_strategy()


    
    def collect_samples(self, history, turn, nn1, nn2):
        pass
        



