from DCFR.DeepCFR import DCFR
from environment.HULH import HULH_Emulator

if __name__ == '__main__':
    env = HULH_Emulator('P1', 'P2')
    p = DCFR(env)
    p.iterate(50, 200, checkpoints=10, verbose_timestep={'i':50, 'k_max':2})

