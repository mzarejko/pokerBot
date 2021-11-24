from DCFR.DeepCFR import DCFR
from environment.HULH import HULH_Emulator
from eval.Evaluator import Evaluator
from DCFR.networks import Poker_network
from tensorflow.keras.models import load_model
import os

def train():
    env = HULH_Emulator('P1', 'P2')
    p = DCFR(env)
    p.iterate(50, 250, checkpoints=10)

def eval():
    models = {}
    for path in os.listdir('./models'):
        models[path] = load_model(f'./models/{path}', custom_objects={'loss_func': 
                                                                      Poker_network.loss_func})
    ev = Evaluator(models)
    winners, h2hs = ev.initial_turnament(100)
    ev.make_chart(winners, h2hs)
    
if __name__ == '__main__':
    #train()
    eval()

