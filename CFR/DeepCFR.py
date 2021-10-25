from .networks import Poker_network
from .Memory import Memory
import numpy as np
import sys
from strenum import StrEnum
import time
import logging
from tqdm import tqdm
from .ModTensorboard import ModTensorBoard
from tensorflow.keras.callbacks import TensorBoard
import time
from .Tree_visualizer.Json_creator import Json_creator

logging.basicConfig(filename='./logs.log', level=logging.DEBUG)

sys.path.append('../environment')
from environment.emulator import Poker_limit_Emulator, params


class Players(StrEnum):
    PLAYER = 'player'
    OPPONENT = 'opponent'


class DCFR:

    def __init__(self):
        self.strategy_memory = Memory()
        self.memory = {Players.PLAYER: Memory(),
                       Players.OPPONENT: Memory()}

        self.advantage_net = {Players.PLAYER: Poker_network('linear'),
                              Players.OPPONENT: Poker_network('linear')}
        self.strategy_net = Poker_network('softmax')
        self.env = Poker_limit_Emulator(Players.PLAYER, Players.OPPONENT)
        self.__tree_vis = Json_creator()

    def iterate(self, iterate, k, visual_each_time):
        for i in tqdm(range(iterate), position=0, desc="Iterate", ncols=100):
            state, events = self.env.initial_game()
            if i+1 % visual_each_time == 0:
                show_tree = True
            else:
                show_tree = False

            for traverser in [Players.PLAYER, Players.OPPONENT]:
                for _ in tqdm(range(k), position=1, desc=f"travers {i+1}, {traverser}", ncols=100):
                    
                    self.traverse(state, 
                                  2,
                                  events,
                                  None,
                                  traverser,
                                  show_tree)

                if self.memory[traverser].is_enough_samples():
                    self.advantage_net[traverser].clear_net()
                    info_set, advantages = self.memory[traverser].get_storage()

                    if i+1 % visual_each_time == 0:
                        self.advantage_net[traverser].train_net(info_set, advantages, 
                                                            TensorBoard(log_dir=f'logs/adv/{traverser}/{i}'))
                    else:
                        self.advantage_net[traverser].train_net(info_set, advantages)
                
            if show_tree:
                self.__tree_vis.save_file(f'data.json')
                self.__tree_vis.clear_tree()
        if self.strategy_memory.is_enough_samples():
            info_set, strategies = self.strategy_memory.get_storage()
            self.strategy_net.train_net(info_set, strategies,
                                        TensorBoard(log_dir=f'logs/strategy/{time.time()}'))
            self.strategy_net.save_model(f'model_{time.time()}')

    def __predict_strategy(self, hole, board, hist, turn):
        advantages = self.advantage_net[turn].predict(hole, board, hist).ravel()
        positive_imm_regrets = []
        for imm_regret in advantages:
            positive_imm_regrets.append(max(0, imm_regret))
            
        cumulative_regrets = sum(positive_imm_regrets)
        
        if cumulative_regrets > 0:
            strategy = positive_imm_regrets / cumulative_regrets
        else:
            strategy = np.repeat(1/params.ACTIONS_NUM, params.ACTIONS_NUM)

        return strategy

    def traverse(self, 
                 state, 
                 timestep,
                 events,
                 previous_turn,
                 traverser,
                 show_tree=False):

        if self.env.is_terminal(state):
            reward = self.env.get_reward(state, events, previous_turn)
            if show_tree:
                self.__tree_vis.update_data(events, timestep, reward*-1)
            return reward

        if self.env.get_turn(state) == traverser:
            action_utils = np.zeros([params.ACTIONS_NUM])
            strategy = self.__predict_strategy(self.env.get_hole_cards(state, traverser), 
                                               self.env.get_community_cards(state),
                                               self.env.get_bet_history(events),
                                               traverser)
            
            for id, action in enumerate(self.env.get_legal_actions(state)):
                new_state, new_events = self.env.act(state, action)
                    
                action_utils[id] = -1 * self.traverse(new_state, 
                                                      timestep+1,
                                                      new_events, 
                                                      self.env.get_turn(state),
                                                      traverser,
                                                      show_tree)

            util = sum(action_utils*strategy)
            if show_tree:
                self.__tree_vis.update_data(events, timestep, util)
            regrets = action_utils - util
            self.collect_samples(self.env.get_bet_history(events),
                                 self.env.get_hole_cards(state, traverser),
                                 self.env.get_community_cards(state),
                                 traverser,
                                 regrets)

            return util

        else:
            strategy = self.__predict_strategy(self.env.get_hole_cards(state,self.env.get_turn(state)),
                                               self.env.get_community_cards(state),
                                               self.env.get_bet_history(events),
                                               self.env.get_turn(state))

            self.collect_strategies(self.env.get_bet_history(events),
                                    self.env.get_hole_cards(state, self.env.get_turn(state)),
                                    self.env.get_community_cards(state),
                                    strategy)

            action = self.env.sample_action(state, strategy)
            new_state, new_events = self.env.act(state, action) 
            
            return -1 * self.traverse(new_state,
                                      timestep+1,
                                      new_events,
                                      self.env.get_turn(state),
                                      traverser,
                                      show_tree)
                
    def collect_samples(self, bet_history, hole_cards, community_cards, traverser, regrets):
        self.memory[traverser].append(hole_cards, community_cards, bet_history, regrets)

    def collect_strategies(self, bet_history, hole_cards, community_cards, strategy):
        self.strategy_memory.append(hole_cards, community_cards, bet_history, strategy)

        



