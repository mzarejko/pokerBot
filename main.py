from .CFR.DeepCFR import DCFR
from .environment.emulator import Poker_limit_Emulator

if __name__ == '__main__':
    env = Poker_limit_Emulator('player', 'opponent')
    dcfr = DCFR(2000, 50, env)
