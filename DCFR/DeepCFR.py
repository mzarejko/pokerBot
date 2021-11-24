import numpy as np
from tqdm import tqdm
from .Tree_visualizer.Tree_creator import Tree_creator
from .player import Brain 
import time


class DCFR:

    def __init__(self, env):
        self.__player = Brain('linear', env, env.player_uuid)
        self.__opponent = Brain('linear', env, env.opponent_uuid)
        self.__strategy = Brain('softmax', env, 'strategy', memory_size=200_000)
        self.__env = env
        self.__tree_diagram = Tree_creator()

    def __save_tree(self):
        self.__tree_diagram.save_file(f'./DCFR/Tree_visualizer/data_{time.time()}.json')
        self.__tree_diagram.clear_tree()

    def iterate(self, iterate, k_iter, checkpoints=None, verbose_timestep={}):
        for i in tqdm(range(iterate), position=0, desc="Iterate", ncols=100):
            state, events = self.__env.initial_game()

            for traverser in [self.__player, self.__opponent]:
                for k in tqdm(range(k_iter), position=1, desc=f"travers {i+1}, {traverser.uuid}", ncols=100):
                    # number of two actions Small blind + Big blind
                    timestep = 2 
                    if verbose_timestep:
                        if verbose_timestep['i'] == i+1 and verbose_timestep['k_max'] > k:
                            self.__traverse(state, 
                                      events,
                                      traverser,
                                      timestep,
                                      verbose=True)
                            self.__save_tree()
                    else:
                        self.__traverse(state, 
                                      events,
                                      traverser,
                                      timestep)

                traverser.train_net()

            if (i+1) % checkpoints == 0:
                self.__strategy.train_net(verbose=True)
                self.__strategy.save()

    def __calc_strategy(self, hole, board, hist, traverser):
        advantages = traverser.predict(hole, board, hist)
        positive_imm_regrets = []
        for imm_regret in advantages:
            positive_imm_regrets.append(max(0, imm_regret))
            
        cumulative_regrets = sum(positive_imm_regrets)
        
        if cumulative_regrets > 0:
            strategy = positive_imm_regrets / cumulative_regrets
        else:
            highest_adv = np.argmax(advantages)
            strategy = positive_imm_regrets
            strategy[highest_adv] = 1

        return strategy

    def __traverse(self, 
                 state, 
                 events,
                 traverser,
                 timestep,
                 previous_turn=None,
                 verbose=False):

        if self.__env.is_terminal(state):
            reward = self.__env.get_reward(state, events, previous_turn)

            if verbose:
                self.__tree_diagram.update_data(events, timestep, reward*-1)
            return reward

        if self.__env.get_turn(state) == traverser.uuid:
            action_utils = np.zeros([self.__env.ACTIONS_NUM])
            strategy = self.__calc_strategy(self.__env.get_hole_cards(state, traverser.uuid), 
                                               self.__env.get_community_cards(state),
                                               self.__env.get_bet_history(events),
                                               traverser)
            
            for id, action in enumerate(self.__env.get_legal_actions(state)):
                new_state, new_events = self.__env.act(state, action)
                    
                action_utils[id] = -1 * self.__traverse(new_state, 
                                                      new_events, 
                                                      traverser,
                                                      timestep+1,
                                                      self.__env.get_turn(state),
                                                      verbose)

            util = sum(action_utils*strategy)
            if verbose:
                self.__tree_diagram.update_data(events, timestep, util)

            regrets = action_utils - util
            traverser.collect_samples(self.__env.get_bet_history(events),
                                      self.__env.get_hole_cards(state, traverser.uuid),
                                      self.__env.get_community_cards(state),
                                      timestep,
                                      regrets)
            return util

        else:
            if self.__env.get_turn(state) == self.__player.uuid:
                turn = self.__player
            else:
                turn = self.__opponent
                
            strategy = self.__calc_strategy(self.__env.get_hole_cards(state, turn.uuid),
                                               self.__env.get_community_cards(state),
                                               self.__env.get_bet_history(events),
                                               turn)

            self.__strategy.collect_samples(self.__env.get_bet_history(events),
                                            self.__env.get_hole_cards(state, turn.uuid),
                                            self.__env.get_community_cards(state),
                                            timestep,
                                            strategy)

            action = self.__env.sample_action(state, strategy)
            new_state, new_events = self.__env.act(state, action) 
            util =  self.__traverse(new_state,
                                    new_events,
                                    traverser,
                                    timestep+1,
                                    self.__env.get_turn(state),
                                    verbose)
            if verbose:
                self.__tree_diagram.update_data(events, timestep, util)
            return -util
