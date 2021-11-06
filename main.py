from DCFR.DeepCFR import DCFR
from environment.HULH import HULH_Emulator

if __name__ == '__main__':
    env = HULH_Emulator('P1', 'P2')
    p = DCFR(env)
    p.iterate(10, 1_000, 2)

