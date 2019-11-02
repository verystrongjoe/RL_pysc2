from absl import app
from absl import flags
import sys
import torch
from utils import arglist
from runs.minigame import MiniGame, MiniGameParallel
from utils.preprocess import Preprocess
import os
from torch.optim import Adam
import torch.multiprocessing as mp

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(arglist.SEED)
FLAGS = flags.FLAGS

flags.DEFINE_string('baseline', 'a3c', 'algorithm')
flags.DEFINE_string('map', 'CollectMineralShards', 'name of game map')
flags.DEFINE_integer('envs', 2, 'number of environments for parallel learning')
flags.DEFINE_integer('resolution', 32, 'resolution for screen and minimap feature layers')
flags.DEFINE_integer('step_mul', 8, 'game steps per agent step')
flags.DEFINE_bool('render', False, 'if true, render the game')
flags.DEFINE_integer('steps_per_batch', 2000, 'number of game batch to train')
flags.DEFINE_float('max_steps', 200000, 'total steps for training')
flags.DEFINE_integer('save_interval', 50, 'number of  steps of saving')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_float('discount', 0.99, 'reward discount factor')
flags.DEFINE_float('entropy_weight', 1e-2, 'weight of entropy loss')
flags.DEFINE_float('value_weight', 0.5, 'weight of value function loss')
flags.DEFINE_bool('has_gpu', True, 'if true, use gpu to run')
flags.DEFINE_string('gpus', "1", 'number of cpus used for training')
flags.DEFINE_string('save_dir', os.path.join('model'), 'model storage')
flags.DEFINE_string('summary_dir', os.path.join('summary'), 'summary storage')
flags.DEFINE_string('model_name', 'my_agent', 'model name')
flags.DEFINE_string('exploration_mode', 'original_ac3', 'if original_ac3, use original a3c mode, else use epsilon-greedy exploration')
flags.DEFINE_bool('set_learning_rate_decay_threshold', True, 'use learning rate decay threshold to accelerate learning efficiency')

FLAGS(sys.argv)

if FLAGS.has_gpu:
	ENVS = FLAGS.envs
	DEVICE = ['/gpu:' + dev for dev in FLAGS.gpus]
else:
	ENVS = 1
	DEVICE = ['/cpu:0']

env_names = ["DefeatZerglingsAndBanelings", "DefeatRoaches",
             "CollectMineralShards", "MoveToBeacon", "FindAndDefeatZerglings",
             "BuildMarines", "CollectMineralsAndGas"]

def main(_):
    for map_name in env_names:
        if FLAGS.baseline == 'ddpg':
            from agent.ddpg import DDPGAgent
            from networks.acnetwork_q_seperated import ActorNet, CriticNet
            from utils.memory import SequentialMemory

            actor = ActorNet()
            critic = CriticNet()
            memory = SequentialMemory(limit=arglist.DDPG.memory_limit)
            learner = DDPGAgent(actor, critic, memory)

            preprocess = Preprocess()
            game = MiniGame(map_name, learner, preprocess, nb_episodes=10000)
            game.run_ddpg()

        elif FLAGS.baseline == 'ppo':
            from agent.ppo import PPOAgent
            from networks.acnetwork_v_seperated import ActorNet, CriticNet
            from utils.memory import EpisodeMemory

            actor = ActorNet()
            critic = CriticNet()
            memory = EpisodeMemory(limit=arglist.PPO.memory_limit,
                                   action_shape=arglist.action_shape,
                                   observation_shape=arglist.observation_shape)
            learner = PPOAgent(actor, critic, memory)

        elif FLAGS.baseline == 'a3c':
            print('start a3c')

            from networks.acnetwork_a3c import A3CNet
            gnet = A3CNet()
            gnet.share_memory()  # share global parameters in multiprocessing
            gnet.to(arglist.DEVICE)
            opt = Adam(gnet.parameters(), lr=0.0002)  # global optimizer
            global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
            preprocess = Preprocess()
            mapname = 'DefeatRoaches'

            # parallel training
            workers = [MiniGameParallel(mapname, preprocess, gnet, opt, global_ep, global_ep_r, res_queue, i,
                             nb_episodes=10000) for i in range(mp.cpu_count())]

            # workers = [MiniGameParallel(map_name, preprocess, gnet, opt, global_ep, global_ep_r, res_queue, i,
            #                                              nb_episodes=10000) for i in range(2)]

            print('created...')

            [w.start() for w in workers]

            print('started...')

            res = []

            while True:
                r = res_queue.get()
                if r is not None:
                    res.append(r)
                else:
                    break

            [w.join() for w in workers]

            import matplotlib.pyplot as plt
            plt.plot(res)
            plt.ylabel('Moving average ep reward')
            plt.xlabel('Step')
            plt.show()

            sys.exit(0)

        else:
            raise NotImplementedError()

        return 0

if __name__ == '__main__':
    app.run(main)
