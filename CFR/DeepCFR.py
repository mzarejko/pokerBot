from enum import Enum
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
        
        self.strategy_memory = Memory()
        self.memory = {Players.PLAYER: Memory(),
                       Players.OPPONENT: Memory()}

        self.advantage_net = {Players.PLAYER: Poker_network(self.BET_HISTORY_LENGTH),
                              Players.OPPONENT: Poker_network(self.BET_HISTORY_LENGTH)}
        self.strategy_net = Poker_network(self.BET_HISTORY_LENGTH)
        
        self.ITERATIONS = iterations
        self.K = k
        self.env = Poker_limit_Emulator(Players.PLAYER, Players.OPPONENT)

    def train(self):
        for _ in self.ITERATIONS:
            state = self.env.initial_game()
            state = self.env.choose_cards(state)
            for p in [Players.PLAYER, Players.OPPONENT]:
                for k in self.K:
                    bet_history = np.zeros([self.BET_HISTORY_LENGTH])
                    self.traverse(bet_history, state, 0, self.env.get_hole_cards(p),
                                  self.env.get_community_cards(), p)
    
    def __predict_strategy(self, hole, board, hist, turn):
        advantages = self.advantage_net[turn].predict(hole, board, hist)
        positive_imm_regrets = []
        for imm_regret in advantages:
            positive_imm_regrets.append(max(0, imm_regret))
            
        cumulative_regrets = sum(positive_imm_regrets)
        
        if cumulative_regrets > 0:
            strategy = positive_imm_regrets / cumulative_regrets
        else:
            strategy = np.zeros([params.ACTIONS_NUM])
            strategy[np.argmax(advantages)] = 1

        return strategy

    def traverse(self, 
                 bet_history,  
                 state, 
                 timestemp,
                 hole_cards,
                 community_cards,
                 traverser):

        if self.env.is_terminal():
            return self.env.get_reward()

        if self.env.get_turn() == traverser:
            action_utils = np.zeros([params.ACTIONS_NUM])

            strategy = self.__predict_strategy(hole_cards, 
                                                  community_cards,
                                                  bet_history,
                                                  traverser)
             
            for id, action in enumerate([params.Actions.RAISE,
                                     params.Actions.CALL,
                                     params.Actions.FOLD]):

                new_state, community_cards = self.env.act(state, action)
                if action == params.Actions.RAISE:
                    bet_history[timestemp] = 1

                if traverser == Players.PLAYER:
                    action_utils[id] = -1 * self.traverse(bet_history, new_state, timestemp+1,
                                                          hole_cards, community_cards,
                                                          Players.OPPONENT)
                elif traverser == Players.OPPONENT:
                    action_utils[id] = -1 * self.traverse(bet_history, new_state, timestemp+1,
                                                          hole_cards, community_cards,
                                                          Players.PLAYER)

            util = sum(action_utils*strategy)
            regrets = action_utils - util
            self.collect_samples(bet_history, hole_cards, community_cards, traverser, regrets)

    def collect_samples(self, history_bet, hole_cards, community_cards, traverser, regrets):
        self.memory[traverser].append((hole_cards, community_cards, history_bet), regrets)
        



