from itertools import combinations
import pandas as pd
import numpy as np
from HULH import HULH_Emulator
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from tensorflow.python.keras.engine.functional import Functional

class Evaluator:

    def __init__(self, models):
        self.__models = models

    def get_model_action(self, model, state, hole, board, hist):
        info_set = np.stack((hole, board, hist), axis=-1)
        strategy = model.predict(info_set[np.newaxis, :]).ravel()
        actions = self.__env.get_legal_actions(state)
        return np.random.choice(actions, p=strategy)
    
    def get_object_action(self, obj, legal_act, hole, board):
        return obj.get_action(legal_act, 
                              self.get_cards(hole), 
                              self.get_cards(board))

    def get_cards(self, one_hot):
        cards = []
        for card in self.__env.CARDS:
            if self.__env.CARDS.index(card) in [id for id, x in enumerate(one_hot) if x == 1]:
                cards.append(card)
        return cards

    def make_action_dist(self, actions, models):
        calls = []
        folds = []
        raises = []

        for idx in range(len(models)):
            total = actions[0][idx] + actions[1][idx] + actions[2][idx]
            raises.append(actions[0][idx]/total*100)
            calls.append(actions[1][idx]/total*100)
            folds.append(actions[2][idx]/total*100)

        df = pd.DataFrame({'calls': calls, 'folds': folds, 'raises': raises},
                          index=models)
        ax = df.plot.bar()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('modele', fontsize=12)
        ax.set_ylabel('Wykonywane akcje [%]', fontsize=12)
        ax.grid(ls=":")
        ax.legend()
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


    def make_chart_average_rewards_lost(self, rews):
        avg = {}
        for model in rews.keys():
            avg[model] = np.mean([-x for x in rews[model] if x < 0])

        id = list(avg.keys())
        data = list(avg.values())

        _, ax = plt.subplots()
        ax.bar(id, data, width=0.4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(ls=":")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        l = ax.get_xticklabels()
        plt.setp(l)
        plt.xlabel('nazwa modeli', fontsize=12)
        plt.ylabel('średnia wartość przegrywanej puli', fontsize=12)
        plt.title('Wyniki uśrednionych pul przegrywanych przez modele\n', fontsize=14)
        plt.show()

    def make_chart_average_rewards_win(self, rews):
        avg = {}
        for model in rews.keys():
            avg[model] = np.mean([x for x in rews[model] if x > 0])

        id = list(avg.keys())
        data = list(avg.values())

        _, ax = plt.subplots()
        ax.bar(id, data, width=0.4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(ls=":")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        l = ax.get_xticklabels()
        plt.setp(l)
        plt.xlabel('nazwa modeli', fontsize=12)
        plt.ylabel('średnia wartość wygrywanej puli', fontsize=12)
        plt.title('Wyniki uśrednionych pul wygrywanych przez modele\n', fontsize=14)
        plt.show()

    def make_games_chart(self, winners, matchs):
        df = pd.DataFrame({'P1': winners[0], 'P2': winners[1]}, index=matchs)
        ax = df.plot.barh()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('wyniki', fontsize=12)
        ax.set_ylabel('kombinacje rozgrywek', fontsize=12)
        ax.grid(ls=":")
        ax.legend()
        ax.get_legend().remove()
        ax.set_title('Wyniki rozgrywek modeli rozpoznawania\n', fontsize=14)
        plt.tight_layout()
        plt.show()


    def initial_turnament(self, iters, h2hs, verbose=False):
        if verbose:
            logging.basicConfig(filename='log.log', level=logging.DEBUG)
        winners = np.zeros([2, len(list(h2hs))])
        matchs = []
        rewards = defaultdict(lambda : [])
        actions_rate = np.zeros([3, len(self.__models)])


        for id, players in enumerate(h2hs):
            matchs.append(f'{players[1]}\n{players[0]}')

            for _ in tqdm(range(iters)):

                self.__env = HULH_Emulator(players[1], players[0], 10)
                game_state, events = self.__env.initial_game()

                rew_total = 0
                while not self.__env.is_end_game(events):
                    if verbose:
                        logging.info('NEW ROUND')
                    while not self.__env.is_terminal(game_state):
                        turn = self.__env.get_turn(game_state)

                        if isinstance(self.__models[turn], Functional):
                            action = self.get_model_action(self.__models[turn], 
                                                    game_state, 
                                                    self.__env.get_hole_cards(game_state, turn),
                                                    self.__env.get_community_cards(game_state),
                                                    self.__env.get_bet_history(events))
                        else:
                            action = self.get_object_action(self.__models[turn],
                                                        self.__env.get_legal_actions(game_state),
                                                        self.__env.get_hole_cards(game_state, turn),
                                                        self.__env.get_community_cards(game_state))
                        if verbose:
                            logging.info(f'Gracz {self.__env.get_turn(game_state)} wykonuje akcję '+
                                        f'{action} przy kartach w ręce '+
                                        f'{self.get_cards(self.__env.get_hole_cards(game_state, turn))} i kartach na '+
                                        f'stole {self.get_cards(self.__env.get_community_cards(game_state))}')

                        game_state, events = self.__env.act(game_state, action)

                        if action == 'raise':
                            actions_rate[0][list(self.__models.keys()).index(turn)] += 1
                        elif action == 'call':
                            actions_rate[1][list(self.__models.keys()).index(turn)] += 1
                        elif action == 'fold':
                            actions_rate[2][list(self.__models.keys()).index(turn)] += 1

                    if not self.__env.is_end_game(events):
                        winner = events[-1]['winners'][0]['uuid']
                        if winner == players[0]:
                            rew = self.__env.get_reward(game_state, events, players[1])
                            rewards[players[0]].append(rew)
                            rewards[players[1]].append(-rew)
                        else:
                            rew = self.__env.get_reward(game_state, events, players[0])
                            rewards[players[1]].append(rew)
                            rewards[players[0]].append(-rew)

                        if verbose:
                            logging.info(f'Rundę zwycięża gracz {winner} z pulą {rew_total}')
                        game_state, events = self.__env.new_round(game_state)
                    else:
                        winner = events[-2]['winners'][0]['uuid']
                        if verbose:
                            logging.info(f'Grę zwycięża gracz {winner} z pulą {rew_total}\n')

                if winner == players[0]:
                    winners[0][id] += 1
                else:
                    winners[1][id] += 1

        return winners, matchs, rewards, actions_rate
