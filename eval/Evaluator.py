from itertools import permutations
import logging 
import matplotlib.pyplot as plt
import numpy as np
from environment.HULH import HULH_Emulator


class Evaluator:

    def __init(self, models):
        self.__models = models
        self.__initial_turnament()

    def get_action(self, model, state, hole, board, hist):
        strategy = model.predict([hole, board, hist]).ravel()
        actions = self.__env.get_legal_actions(state)
        return np.random.choice(actions, p=strategy)

    def get_cards(self, one_hot):
        cards = []
        for card in self.__env.CARDS:
            if self.__env.CARDS.index(card) in [x for x in one_hot if x == 1]:
                cards.append(card)
        return cards
                
    def __initial_turnament(self):
        h2hs = permutations(self.__models.keys())

        for player in h2hs:
            logging.basicConfig(filename=f'./matchs/{player[0]}_vs_{player[1]}',
                                level=logging.DEBUG)

            self.__env = HULH_Emulator(player[0], player[1])
            game_state, events = self.__env.initial_game()
            logging.info('INITIAL GAME')
            logging.info(f'Player {player[0]}: \n'
                         f'cards: {self.get_cards(self.__env.get_hole_cards(game_state, player[0]))}')
            logging.info(f'Player {player[1]}: \n'
                         f'cards: {self.get_cards(self.__env.get_hole_cards(game_state, player[1]))}')

            prev_turn = None
            while not self.__env.is_terminal(game_state):
                action = self.get_action(self.__models[self.__env.get_turn(game_state)], 
                                         game_state, 
                                         self.__env.get_hole_cards(game_state, player[0]),
                                         self.__env.get_community_cards(game_state),
                                         self.__env.get_bet_history(events))

                logging.info(f'Player {self.__env.get_turn(game_state)} {action}')
                logging.info(f'BOARD CARDS {self.get_cards(self.__env.get_community_cards(game_state))}')
                prev_turn = self.__env.get_turn(game_state)
                game_state, events = self.__env.act(game_state, action)

            rew = self.__env.get_reward(game_state, events, prev_turn)
            logging.info(f'Player {prev_turn} get {rew}')











        

        
