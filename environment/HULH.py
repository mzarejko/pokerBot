from pypokerengine.api.emulator import Emulator
import random
import numpy as np
from pypokerengine.utils.game_state_utils import attach_hole_card_from_deck
from collections import deque


class HULH_Emulator:


    def __init__(self, player_uuid, opponent_uuid):
        self.emulator = Emulator()
        self.player_uuid = player_uuid
        self.opponent_uuid = opponent_uuid

        self.SMALL_BLIND = 5
        self.ROUNDS_LIMIT = 1 
        self.ANTE = 5
        self.NUM_PLAYERS = 2
        self.BIG_BLIND = 10
        self.STACK = 100
        self.CARDS = ['C2', 'D2', 'H2', 'S2',
                      'C3', 'D3', 'H3', 'S3',
                      'C4', 'D4', 'H4', 'S4',
                      'C5', 'D5', 'H5', 'S5',
                      'C6', 'D6', 'H6', 'S6',
                      'C7', 'D7', 'H7', 'S7',
                      'C8', 'D8', 'H8', 'S8',
                      'C9', 'D9', 'H9', 'S9',
                      'CT', 'DT', 'HT', 'ST',
                      'CJ', 'DJ', 'HJ', 'SJ',
                      'CQ', 'DQ', 'HQ', 'SQ',
                      'CK', 'DK', 'HK', 'SK',
                      'CA', 'DA', 'HA', 'SA']
        self.ACTIONS_NUM = 3
        self.BET_HISTORY_LENGTH = 52


    def initial_game(self):
        self.emulator.set_game_rule(player_num=self.NUM_PLAYERS,
                               max_round=self.ROUNDS_LIMIT,
                               small_blind_amount=self.SMALL_BLIND,
                               ante_amount=self.ANTE)
        
        first_reg = random.choice([self.player_uuid, self.opponent_uuid]) 

        if first_reg == self.player_uuid:
            players_info = {
                self.player_uuid: { "name": "player", "stack": self.STACK},
                self.opponent_uuid: { "name": "opponent", "stack": self.STACK}
            }
        else:
            players_info = {
                self.opponent_uuid: { "name": "opponent", "stack": self.STACK },
                self.player_uuid: { "name": "player", "stack": self.STACK }
            }

        initial_state = self.emulator.generate_initial_game_state(players_info)
        game_state, events = self.emulator.start_new_round(initial_state)
        for player in game_state['table'].seats.players:
            game_state = attach_hole_card_from_deck(game_state, player.uuid)
            
        return game_state, events

    def __convert_to_one_hot_cards(self, cards):
        one_hot = np.zeros([len(self.CARDS)])
        for card in cards:
            idx = self.CARDS.index(str(card))
            one_hot[idx] = 1

        return one_hot

    def get_hole_cards(self, state, uuid):
        players = state["table"].seats.players 
        for player in players:
            if player.uuid == uuid:
                return self.__convert_to_one_hot_cards([str(player.hole_card[0]),
                                                  str(player.hole_card[1])])
        raise Exception("error: uuid not exits: ", uuid)

    def get_reward(self, state, events, prev_turn):
        if self.is_terminal(state):
            winner = events[-2]['winners'][0]['uuid']
            if prev_turn == winner:
                return -1 * events[0]['round_state']['pot']['main']['amount']
            else:
                return events[0]['round_state']['pot']['main']['amount']
  
    def get_bet_history(self, events):
        bet_history = deque(maxlen=self.BET_HISTORY_LENGTH)
        history = events[0]["round_state"]["action_histories"]
        for r in history.keys():
            for action in history[r]:
                if action['action'] == "RAISE":
                    bet_history.append(1)
                else:
                    bet_history.append(0)
                    
        bet_list = np.zeros([self.BET_HISTORY_LENGTH])
        for id, bet in enumerate(bet_history):
            bet_list[id] = bet
            
        return bet_list 
            
    def get_turn(self, state):
        return state["table"].seats.players[state["next_player"]].uuid

    def is_terminal(self, state):
        return state['street'] == 5

    def sample_action(self, state, prob):
        action =  np.random.choice(self.get_legal_actions(state), p=prob) 
        return action

    def get_community_cards(self, state):
        cards = state['table'].get_community_card()
        return self.__convert_to_one_hot_cards(cards)

    def get_legal_actions(self, state):
        actions = []
        act_list = self.emulator.generate_possible_actions(state)
        for act in act_list:
            actions.append(act['action'])
        return actions

    def get_legal_amount(self, state, act):
        act_list = self.emulator.generate_possible_actions(state)
        for id, a in enumerate(act_list):
            if a['action'] == act:
                if act == 'raise':
                    return act_list[id]['amount']['min']
                else:
                    return act_list[id]['amount']

    def act(self, state, action):
        amount = self.get_legal_amount(state, action)
        if action:
            state, events = self.emulator.apply_action(state, action, amount)
        else:
            raise Exception('invalid action: ', action)  
        return state, events

