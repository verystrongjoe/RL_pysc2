import time
import os
from pysc2.env import sc2_env
from utils import arglist
from copy import deepcopy
import torch.multiprocessing as mp
import numpy as np
import torch
from torch.distributions import Categorical
from pysc2.lib import actions

agent_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
        minimap=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), )
)

from networks.acnetwork_a3c import A3CNet

class MiniGameParallel(mp.Process):
    def __init__(self, map_name, preprocess, gnet, opt, global_ep, global_ep_r, res_queue, name, nb_episodes=50000):
        super(MiniGameParallel, self).__init__()
        self.map_name = map_name
        self.nb_max_steps = 200
        self.nb_episodes = nb_episodes
        self.preprocess = preprocess
        self.lnet = A3CNet()
        self.name = 'w%i'% name # todo
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt     # todo: what is opt?
        self.env = sc2_env.SC2Env(map_name=self.map_name,
                                  step_mul=8,
                                  visualize=False,
                                  game_steps_per_episode=1000,
                                  agent_interface_format=[agent_format])

    def preprocess_available_actions(self, available_actions, max_action=arglist.NUM_ACTIONS):
        a_actions = np.zeros(max_action, dtype='float32')
        a_actions[available_actions] = 1.
        return a_actions

    def _mask_unavailable_actions(self, policy, valid_actions):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        valid_actions = torch.from_numpy(valid_actions).to(arglist.DEVICE)
        masked_policy_vb = policy * valid_actions
        # masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb

    def _test_valid_action(self, function_id, valid_actions):
        if valid_actions[function_id] == 1:
            return True
        else:
            return False

    def select_action(self, obs, valid_actions):
        '''
        from logit to pysc2 actions
        :param logits: {'categorical': [], 'screen1': [], 'screen2': []}
        :return: FunctionCall form of action
        '''
        obs_torch = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for o in obs:
            x = obs[o].astype('float32')
            x = np.expand_dims(x, 0)
            obs_torch[o] = torch.from_numpy(x).to(arglist.DEVICE)

        logits = self.actor(obs_torch)
        prob_categorical = torch.nn.Softmax(dim=-1)(logits['categorical']).detach()
        prob_categorical = self._mask_unavailable_actions(prob_categorical, valid_actions)
        dist = Categorical(prob_categorical)

        try:
            function_id = dist.sample().item()
        except RuntimeError:
            function_id = 0

        is_valid_action = self._test_valid_action(function_id, valid_actions)

        # todo: 여기확률을 0으로 만드는데 굳이 validate 할 필요가 있는가?
        while not is_valid_action:
            function_id = dist.sample().item()
            is_valid_action = self._test_valid_action(function_id, valid_actions)

        # todo : 여기같은 경우에도 좌표를 아무래도 상대좌표라고 해도 범위를 제한하는게 좋지 않는감?
        p = torch.nn.Softmax(dim=-1)(logits['screen1'].view(1, -1)).detach()
        pos_screen1 = Categorical(p).sample().item()
        p = torch.nn.Softmax(dim=-1)(logits['screen2'].view(1, -1)).detach()
        pos_screen2 = Categorical(p).sample().item()

        # todo : range를 좀 더 줄이는게 의미가 있지 않을까?
        pos = [[int(pos_screen1 % arglist.FEAT2DSIZE), int(pos_screen1 // arglist.FEAT2DSIZE)],
               [int(pos_screen2 % arglist.FEAT2DSIZE), int(pos_screen2 // arglist.FEAT2DSIZE)]]  # (x, y)

        args = []
        cnt = 0
        for arg in actions.FUNCTIONS[function_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(pos[cnt])
                cnt += 1
            else:
                args.append([0])

        action = actions.FunctionCall(function_id, args)
        return action

    def run(self):
        self.env = sc2_env.SC2Env(map_name=self.map_name,
                                  step_mul=8,
                                  visualize=False,
                                  game_steps_per_episode=1000,
                                  agent_interface_format=[agent_format])

        while self.g_ep.value < self.nb_episodes:
            s = self.env.reset()[0]

            buffer_s, buffer_a, buffer_r = [],[],[]
            ep_r = 0.

            for t in range(self.nb_max_steps):  # todo : 이값은 우찌 설정?

                obs = self.preprocess.get_observation(s)
                actions = self.select_action(obs, valid_actions=obs['available_actions'])
                state_new = self.env.step(actions=[actions])[0]
                actions = self.preprocess.postprocess_action(actions)

                if s.last():
                    cum_reward = s.observation["score_cumulative"][0]
                    #save network
                    break
                else:
                    state = deepcopy(state_new)

            self.res_queue.put(None)
        self.env.close()

class MiniGame:
    def __init__(self, map_name, learner, preprocess, nb_episodes=50000):
        self.map_name = map_name
        self.nb_max_steps = 200
        self.nb_episodes = nb_episodes
        self.env = sc2_env.SC2Env(map_name=self.map_name,
                                  step_mul=8,
                                  visualize=False,
                                  game_steps_per_episode=1000,
                                  agent_interface_format=[agent_format])
        self.learner = learner
        self.preprocess = preprocess

    def write_history(self, fname, msg=None):
        fname = 'Models/' + fname

        if not os.path.exists(fname) or msg is None:
            f = open(fname, "w")
        else:
            f = open(fname, "a")
        f.write(str(msg) + '\n')
        f.close()

    def run_ddpg(self, is_training=True):
        cum_reward_best = 0
        self.write_history(self.map_name + '_history_ddpg.txt', msg=None)

        for i_episode in range(self.nb_episodes):
            state = self.env.reset()[0]
            for t in range(self.nb_max_steps):  # Don't infinite loop while learning
                obs = self.preprocess.get_observation(state)
                actions = self.learner.select_action(obs, valid_actions=obs['nonspatial'])
                state_new = self.env.step(actions=[actions])[0]
                # print(actions)
                # append memory
                actions = self.preprocess.postprocess_action(actions)
                self.learner.memory.append(obs, actions, state.reward, state.last(), training=is_training)

                self.learner.optimize(is_train=self.learner.iter % 8 == 0)

                if state.last():
                    cum_reward = state.observation["score_cumulative"][0]
                    # save networks
                    if cum_reward >= cum_reward_best:
                        self.learner.save_models(fname=self.map_name + '_ddpg')
                        cum_reward_best = cum_reward

                    # write cululative reward for every episodes
                    self.write_history(fname=self.map_name + '_history_ddpg.txt',
                                       msg='episde: {}, step: {}, score: {}'.format(i_episode, self.learner.iter, cum_reward))
                    break
                else:
                    state = deepcopy(state_new)

                self.learner.iter += 1
        self.env.close()

    def run_ppo(self, is_training=True):
        reward_cumulative = []
        f = open("PPO_result.txt", "w")
        for i_episode in range(self.nb_episodes):
            state = self.env.reset()[0]
            for t in range(self.nb_max_steps):  # Don't infinite loop while learning
                obs = self.preprocess.get_observation(state)
                actions = self.learner.select_action(obs, valid_actions=obs['nonspatial'])
                state_new = self.env.step(actions=[actions])[0]

                # append memory
                actions = self.preprocess.postprocess_action(actions)
                self.learner.memory.append(obs, actions, state.reward, state.last(), training=is_training)

                if state.last():
                    f = open("PPO_result.txt", "a")
                    cum_reward = state.observation["score_cumulative"]
                    reward_cumulative.append(cum_reward[0])
                    start = time.time()
                    self.learner.optimize(update=True)
                    self.learner.memory.clear()
                    end = time.time()
                    print(end-start)
                    f.write(f"score: [{cum_reward[0]}]\n")
                    break
                else:
                    state = deepcopy(state_new)

            time.sleep(0.5)

            if (i_episode + 1) % 10000 == 0:
                self.learner.save_models(fname=i_episode)

        self.env.close()
        f.close()
        print(reward_cumulative)

