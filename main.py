from DCFR.DeepCFR import DCFR
from HULH import HULH_Emulator
from Evaluator import Evaluator
from DCFR.networks import Poker_network
from tensorflow.keras.models import load_model
from HonestPlayer import HonestPlayer
from itertools import combinations

def train():
    env = HULH_Emulator('P1', 'P2', 5)
    p = DCFR(env)
    p.iterate(50, 270, checkpoints=10)

def eval(paths, iters):
    models = {}
    for path in paths:
        models[path.split('_')[-1]] = load_model(path, custom_objects={'loss_func': Poker_network.loss_func})

    h2hs = list(combinations(models.keys(), 2))
    ev = Evaluator(models)
    winners, h2hs, rews, actions = ev.initial_turnament(iters, h2hs)
    ev.make_games_chart(winners, h2hs)
    ev.make_chart_average_rewards_lost(rews)
    ev.make_chart_average_rewards_win(rews)
    ev.make_action_dist(actions, list(models.keys()))

def model_vs_honest(paths, iters):
    models = {}
    player = HonestPlayer(100, 2)
    models['HP'] = player
    for path in paths:
        models[path.split('_')[-1]] = load_model(path, custom_objects={'loss_func': Poker_network.loss_func})
    h2hs = list(combinations(models.keys(), 2))
    for vs in h2hs:
        if 'HP' not in vs:
            h2hs.remove(vs)

    ev = Evaluator(models)
    winners, h2hs, rews, actions= ev.initial_turnament(iters, h2hs, verbose=True)
    ev.make_games_chart(winners, h2hs)
    ev.make_chart_average_rewards_lost(rews)
    ev.make_chart_average_rewards_win(rews)
    ev.make_action_dist(actions, list(models.keys()))
    
if __name__ == '__main__':
    #train()
    eval(['./models/model_1', 
           './models/model_2',
           './models/model_3',
           './models/model_4',
           './models/model_5']
          , 1)
    # model_vs_honest(['./models/model_1', './models/model_3'], 200)
