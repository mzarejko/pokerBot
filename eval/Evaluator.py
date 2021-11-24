from itertools import permutations
import pandas as pd
import numpy as np
from environment.HULH import HULH_Emulator
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


class Evaluator:

    def __init__(self, models):
        self.__models = models

    def get_action(self, model, state, hole, board, hist):
        info_set = np.stack((hole, board, hist), axis=-1)
        strategy = model.predict(info_set[np.newaxis, :]).ravel()
        actions = self.__env.get_legal_actions(state)
        return np.random.choice(actions, p=strategy)

    def get_cards(self, one_hot):
        cards = []
        for card in self.__env.CARDS:
            if self.__env.CARDS.index(card) in [id for id, x in enumerate(one_hot) if x == 1]:
                cards.append(card)
        return cards

    def make_chart(self, winners, matchs):
        df = pd.DataFrame({'P1': winners[0], 'P2': winners[1]}, index=matchs)
        ax = df.plot.barh()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('wyniki', fontsize=8)
        ax.set_ylabel('mecze', fontsize=8)
        ax.grid(ls=":")
        ax.legend()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        ax.set_title('Wyniki meczy modeli rozpoznawania')
        plt.tight_layout()
        plt.show()

    def initial_turnament(self, iters):
        h2hs = list(permutations(self.__models.keys(), 2))
        winners = np.zeros([2, len(list(h2hs))])
        matchs = []

        print()
        for id, players in enumerate(h2hs):
            matchs.append(f'{players[0]} vs {players[1]}')

            for _ in tqdm(range(iters)):

                self.__env = HULH_Emulator(players[0], players[1])
                game_state, events = self.__env.initial_game()

                while not self.__env.is_terminal(game_state):
                    action = self.get_action(self.__models[self.__env.get_turn(game_state)], 
                                             game_state, 
                                             self.__env.get_hole_cards(game_state, players[0]),
                                             self.__env.get_community_cards(game_state),
                                             self.__env.get_bet_history(events))

                    game_state, events = self.__env.act(game_state, action)

                winner = events[-2]['winners'][0]['uuid']
                if winner == players[0]:
                    winners[0][id] += 1
                else:
                    winners[1][id] += 1

        return winners, matchs











        

        
