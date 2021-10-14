from enum import Enum

SMALL_BLIND = 5
ROUNDS_LIMIT = 1 
ANTE = 0
NUM_PLAYERS = 2
BIG_BLIND = 10
STACK = 100
CARDS = ['C2', 'D2', 'H2', 'S2',
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
NUM_HOLE_CARDS = 2
NUM_BOARD_CARDS = 5
FLOP_CARDS = 3
TURN_CARDS = 1
RIVER_CARDS = 1
ACTIONS_NUM = 3

class Actions(str, Enum):
    RAISE = 'raise'
    FOLD = 'fold w'
    CALL = 'call'
