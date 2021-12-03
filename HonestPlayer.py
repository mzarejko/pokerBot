from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

class HonestPlayer:

    def __init__(self, nb_simulations, nb_players):
        self.__nb_simulations = nb_simulations
        self.__nb_players = nb_players

    def get_action(self, legal_actions, hole_cards, community_cards):
        win_rate = estimate_hole_card_win_rate(nb_simulation=self.__nb_simulations,
                                               nb_player=self.__nb_players,
                                               hole_card=gen_cards(hole_cards),
                                               community_card=gen_cards(community_cards))
        if win_rate >= 1.0 / self.__nb_players and 'call' in legal_actions:
            action = 'call' 
        elif win_rate < 1.0/ self.__nb_players and 'fold' in legal_actions:
            action = 'fold'
        else:
            raise Exception('Error, legal actions do not have call and fold')
        return action




