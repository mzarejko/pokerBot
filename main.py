from DCFR.DeepCFR import DCFR
from environment.HULH import HULH_Emulator
from .eval.Evaluator import Evaluator
from tensorflow.keras.models import load_model
import os

def train():
    env = HULH_Emulator('P1', 'P2')
    p = DCFR(env)
    p.iterate(50, 200, checkpoints=10, verbose_timestep={'i':50, 'k_max':2})

def eval():
    models = {}
    for path in os.listdir('./models'):
        models[path] = load_model(f'./models/{path}')
    ev = Evaluator(models)
    
if __name__ == '__main__':
    #train()
    eval()

