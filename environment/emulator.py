from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import attach_hole_card
from pypokerengine.utils.card_utils import gen_cards
import random
import logging
import numpy as np
from . import params

logging.basicConfig(filename='./logs.log', level=logging.DEBUG)


class Poker_limit_Emulator:


    def __init__(self, player_uuid, opponent_uuid):
        self.emulator = Emulator()
        self.player_uuid = player_uuid
        self.opponent_uuid = opponent_uuid

        # const params
        
        # min amount money to start street
        self.__offset = params.SMALL_BLIND
        self.__current_event = None
        self.__current_stacks = {player_uuid: params.STACK,
                               opponent_uuid: params.STACK}
        self.__turn = None
        self.__hole_cards = {player_uuid: [],
                             opponent_uuid: []}

    def initial_game(self):
        self.emulator.set_game_rule(player_num=params.NUM_PLAYERS,
                               max_round=params.ROUNDS_LIMIT,
                               small_blind_amount=params.SMALL_BLIND,
                               ante_amount=params.ANTE)
        
        small_blind = random.choice([self.player_uuid, self.opponent_uuid]) 
        self.__turn = small_blind
        self.__comunity_cards = np.zeros([len(params.CARDS)])

        if small_blind == self.player_uuid:
            players_info = {
                self.player_uuid: { "name": "player", "stack": params.STACK},
                self.opponent_uuid: { "name": "opponent", "stack": params.STACK}
            }

            logging.info('Game start \n\n' +
                         f'stack {params.STACK} \nsmall_blind {self.player_uuid}, \n' +
                         f'big_bind {self.opponent_uuid}')
        else:
            players_info = {
                self.opponent_uuid: { "name": "opponent", "stack": params.STACK },
                self.player_uuid: { "name": "player", "stack": params.STACK }
            }

            logging.info('Game start \n\n' +
                         f'stack {params.STACK} \nsmall_blind {self.opponent_uuid}, \n' +
                         f'big_bind {self.player_uuid}')

        initial_state = self.emulator.generate_initial_game_state(players_info)
        state, events = self.emulator.start_new_round(initial_state)
        self.__current_event = events
        return state 

    def __convert_to_one_hot(self, cards):
        one_hot = np.zeros([len(params.CARDS)])
        for card in cards:
            idx = params.CARDS.index(card)
            one_hot[idx] = 1

        return one_hot


    def choose_cards(self, state):
        player_cards = []
        opponent_cards = []
        for _ in range(params.NUM_HOLE_CARDS):
            for player in [player_cards, opponent_cards]:
                card = random.choice(params.CARDS)
                player.append(card)
        
        if player_cards[0] in opponent_cards or player_cards[1] in opponent_cards:
            state, player_cards, opponent_cards = self.choose_cards(state)
        
        logging.info('\nCards dealed: \n\n'+
                     f'{self.player_uuid} {player_cards} \n'+
                     f'{self.opponent_uuid} {opponent_cards}')

        state = self.__deal_player_cards(state, player_cards, opponent_cards)
        self.__hole_cards[self.player_uuid] = self.__convert_to_one_hot(player_cards)
        self.__hole_cards[self.opponent_uuid] = self.__convert_to_one_hot(opponent_cards)
        return state
        
    def __deal_player_cards(self, game_state, player_cards, opponent_cards):
        for player in game_state["table"].seats.players:
            if player.uuid == self.player_uuid:
                hole_cards = gen_cards(player_cards)
                game_state = attach_hole_card(game_state, 
                                              player.uuid, 
                                              hole_cards)
            elif player.uuid == self.opponent_uuid:
                hole_cards = gen_cards(opponent_cards)
                game_state = attach_hole_card(game_state, 
                                              player.uuid, 
                                              hole_cards)
        return game_state

    def get_hole_cards(self, uuid):
        if uuid == self.player_uuid:
            return self.__hole_cards[self.player_uuid]
        elif uuid == self.opponent_uuid:
            return self.__hole_cards[self.opponent_uuid]
        else:
            raise Exception("error: uuid not exits: ", uuid)

    def get_reward(self):
        if self.is_terminal():
            if self.__turn == self.player_uuid:
                reward = self.__current_stacks[self.player_uuid] - self.__current_stacks[self.opponent_uuid]
                logging.info(f'\n{self.player_uuid} reward : {reward}')
            elif self.__turn == self.opponent_uuid:
                reward = self.__current_stacks[self.opponent_uuid] - self.__current_stacks[self.player_uuid]
                logging.info(f'\n{self.opponent_uuid} reward : {reward}')
            else:
                raise Exception('Wrong argument for turn!')

            return reward
    
    def get_turn(self):
        return self.__turn

    def get_winner(self):
        if self.is_terminal():
            self.__offset = params.SMALL_BLIND
            winner = self.__current_event[0]['winners'][0]['uuid']
            logging.info(f'\nwinner : {winner}, stack : '+
                         f'{self.__current_stacks[self.player_uuid]} | '
                         f'{self.__current_stacks[self.opponent_uuid]}')
            return winner

    def __update_game(self, events):
        if events[0]['round_state']['seats'][0]['uuid'] == self.player_uuid:
            self.__current_stacks[self.player_uuid] = events[0]['round_state']['seats'][0]['stack']
            self.__current_stacks[self.opponent_uuid] = events[0]['round_state']['seats'][1]['stack']
        else:
            self.__current_stacks[self.player_uuid] = events[0]['round_state']['seats'][1]['stack']
            self.__current_stacks[self.opponent_uuid] = events[0]['round_state']['seats'][0]['stack']
        
        logging.info(f'{self.player_uuid} {self.__current_stacks[self.player_uuid]} ,'
                     f'{self.opponent_uuid} {self.__current_stacks[self.opponent_uuid]}')

        self.__current_event = events[0]['type']

    def is_terminal(self):
        return self.__current_event[0]['type'] == 'event_round_finish'

    def is_chance(self):
        return self.__current_event[0]['type'] == 'event_new_street'

    def get_street(self):
        return self.__current_event[0]['round_state']['street']

    def sample_action(self):
        return np.random.choice([params.Actions.RAISE,
                                 params.Actions.CALL,
                                 params.Actions.FOLD]) 

    def get_community_cards(self):
        return self.__comunity_cards

    def act(self, state, action):
        # update turn
        if self.__turn == self.player_uuid:
            self.__turn = self.opponent_uuid
        else:
            self.__turn = self.player_uuid
        
        # perform actions
        if action == params.Actions.RAISE:
            state, events = self.emulator.apply_action(state, action,
                                                           params.BIG_BLIND+self.__offset)
            logging.info(f'action : raise {params.BIG_BLIND + self.__offset}')
            self.__offset += params.BIG_BLIND

        elif action == params.Actions.CALL:
            state, events = self.emulator.apply_action(state, action, self.__offset)
            logging.info(f'action : call {self.__offset}')

        elif action == params.Actions.FOLD:
            state, events = self.emulator.apply_action(state, action, 0)
            logging.info(f'action : fold')

        else:
            raise Exception('invalid action: ', action)  
        
        self.__current_event = events
        self.__update_game(events)
        
        community_card = events[0]['round_state']['community_card']
        logging.info(f'community_card : {community_card}\n')
        self.__comunity_cards = self.__convert_to_one_hot(community_card)
        
        return state, self.__comunity_cards
