import numpy as np
from absl import flags
import sys
import os
import shutil
import sys
import argparse
from functools import partial
import torch

# environment
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES
from environ.pre_processing import Preprocessor
from environ.pre_processing import is_spatial_action, stack_ndarray_dicts
from environ.environment import SubprocVecEnv, make_sc2env, SingleEnv
from utils.pre_processing import Preprocessor

# agent
from agent.a2c import A2CAgent
from runs.logger import Logger

FLAGS = flags.FLAGS
FLAGS(['runner.py'])

parser = argparse.ArgumentParser(description='Starcraft 2 deep RL agents')
parser.add_argument('--experiment_id', type=str, required=True,
                    help='identifier to store experiment results')
parser.add_argument('--eval', action='store_true',
                    help='if false, episode scores are evaluated')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing experiments (if --train=True)')
parser.add_argument('--map', type=str, default='MoveToBeacon',
                    help='name of SC2 map')
parser.add_argument('--vis', action='store_true',
                    help='render with pygame')
parser.add_argument('--max_windows', type=int, default=1,
                    help='maximum number of visualization windows to open')
parser.add_argument('--res', type=int, default=32,
                    help='screen and minimap resolution')
parser.add_argument('--envs', type=int, default=2,
                    help='number of environments simulated in parallel')
parser.add_argument('--step_mul', type=int, default=8,
                    help='number of game steps per agent step')
parser.add_argument('--steps_per_batch', type=int, default=16,
                    help='number of agent steps when collecting trajectories for a single batch')
parser.add_argument('--discount', type=float, default=0.95,
                    help='discount for future rewards')
parser.add_argument('--iters', type=int, default=-1,
                    help='number of iterations to run (-1 to run forever)')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device id')
parser.add_argument('--summary_iters', type=int, default=1,
                    help='record summary after this many iterations')
parser.add_argument('--save_iters', type=int, default=500,
                    help='store checkpoint after this many iterations')
parser.add_argument('--max_to_keep', type=int, default=5,
                    help='maximum number of checkpoints to keep before discarding older ones')
parser.add_argument('--entropy_weight', type=float, default=1e-4,
                    help='weight of entropy penalty')
parser.add_argument('--value_loss_weight', type=float, default=1.0,
                    help='weight of value function loss')
parser.add_argument('--max_gradient_norm', type=float, default=500.0,
                    help='Clip gradients')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='Is using cuda or not')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--save_dir', type=str, default='out/models',
                    help='root directory for checkpoint storage')
parser.add_argument('--summary_dir', type=str, default='out/summary',
                    help='root directory for summary storage')
# Bookeeping
args = parser.parse_args()

if not os.path.exists('out'):
    os.mkdir('out')
model_dir = os.path.join('out', 'pre_trained')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

print('Training model file will be created  in {}'.format(model_dir))

args.train = not args.eval
ckpt_path = os.path.join(args.save_dir, args.experiment_id)
summary_type = 'train' if args.train else 'eval'
summary_path = os.path.join(args.summary_dir, args.experiment_id, summary_type)

args.save_dir = ckpt_path
args.summary_dir = summary_path


def compute_returns_advantages(rewards, dones, values, next_values, discount):
    """
    n개의 transition 샘플을 가지고 리턴과 어드벤테이지를 구함.
    받은 리워드와 Critic의 Value값을 가지고 Loss를 계산전에 구함
    Args:
      rewards: array of shape [n_steps, n_env] containing received rewards.
      dones: array of shape [n_steps, n_env] indicating whether an episode is
        finished after a time step.
      values: array of shape [n_steps, n_env] containing estimated values.
      next_values: array of shape [n_env] containing estimated values after the
        last step for each environment.
      discount: scalar discount for future rewards.

    Returns:
      returns: array of shape [n_steps, n_env]
      advs: array of shape [n_steps, n_env]
    """
    returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])
    returns[-1, :] = next_values

    for t in reversed(range(rewards.shape[0])):
        future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
        returns[t, :] = rewards[t, :] + future_rewards

    returns = returns[:-1, :]
    advs = returns - values

    return returns, advs


def mask_unused_argument_samples(actions):
    """
    Replace sampled argument id by -1 for all arguments not used in a steps action (in-place).
    """
    fn_id, arg_ids = actions
    # print('fn_id : {}, arg_ids : {}'.format(fn_id, arg_ids))
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[a_0].args)
        for arg_type in unused_types:
            arg_ids[arg_type][n] = -1
    return fn_id, arg_ids


def save_checkpoint(epoch_count, network, optimizer):
    print('Saving checkpoint at', os.path.join(model_dir, 'model.pth.tar'))
    out = {}
    params = network.get_trainable_params(with_id=True)
    for k, v in params.items():
        out[k] = v.state_dict()
    out['epoch'] = epoch_count
    out['optimizer'] = optimizer.state_dict()  # todo : optimizer needs to be saved?
    torch.save(out, os.path.join(model_dir, 'model.pth.tar'))


def actions_to_pysc2(actions, size):
    """Convert agent action representation to FunctionCall representation."""
    height, width = size
    fn_id, arg_ids = actions
    actions_list = []
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        a_l = []
        for arg_type in FUNCTIONS._func_list[a_0].args:
            arg_id = arg_ids[arg_type][n]
            if is_spatial_action[arg_type]:
                arg = [arg_id % width, arg_id // height]
            else:
                arg = [arg_id]
            a_l.append(arg)
        action = FunctionCall(a_0, a_l)
        actions_list.append(action)
    return actions_list


def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
    return {k: flatten_first_dims(v) for k, v in x.items()}


def stack_and_flatten_actions(lst, axis=0):
    fn_id_list, arg_dict_list = zip(*lst)
    fn_id = np.stack(fn_id_list, axis=axis)
    fn_id = flatten_first_dims(fn_id)
    arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
    arg_ids = flatten_first_dims_dict(arg_ids)
    return fn_id, arg_ids

class A2CRunner(object):
    def __init__(self, agent, envs, summary_writer, train=True, n_steps=8):
        """
         Args:
           agent: A2CAgent instance.
           envs: SubprocVecEnv instance.
           summary_writer: summary writer to log episode scores.
           train: whether to train the agent.
           n_steps: number of agent steps for collecting rollouts.
         """
        self.agent = agent
        self.envs = envs
        self.summary_writer = summary_writer
        self.train = train
        self.n_steps = n_steps
        self.episode_counter = 0
        self.preproc = Preprocessor()
        self.cumulative_score = 0.0
        self.episode_counter = 0
        self.discount = 0.99


    def reset(self):
        obs_raw = self.envs.reset()
        # todo : here, every reset in episode of same scenario could return different available scenario!!
        self.last_observations = self.preproc.preprocess_obs(obs_raw)


    def run_batch(self, train_summary=False):
        """
        Collect trajectories in a single batch just in case of training period
        :param train_summary:
        :return:
        """
        global last_obs
        shapes = (self.n_steps, self.envs.n_envs)
        values = np.zeros(shapes, dtype=np.float32)
        rewards = np.zeros(shapes, dtype=np.float32)
        dones = np.zeros(shapes, dtype=np.float32)

        all_observations = []
        all_actions = []
        all_scores = []

        last_observations = self.last_observations

        for n in range(self.n_steps):
            actions, value_estimates = self.agent.step(last_observations)

            print(sum(last_observations['available_actions'][0]))
            print(sum(last_observations['available_actions'][1]))

            actions = mask_unused_argument_samples(actions)
            size = last_observations['screen'].shape[2:]
            values[n, :] = value_estimates
            all_observations.append(last_observations)
            all_actions.append(actions)

            pysc2_actions = actions_to_pysc2(actions, size)
            obs_raw = self.envs.step(pysc2_actions)
            last_obs = self.preproc.preprocess_obs(obs_raw) # todo : check here whether or not its order is guarranted.
            rewards[n, :] = [t.reward for t in obs_raw]
            dones[n, :] = [t.last() for t in obs_raw]

            for t in obs_raw:
                if t.last():
                    score = self._summarize_episode(t)
                    self.cumulative_score += score
                    # todo : done check!!
                    self.reset()  # reset game without interacting with agent.

        self.last_observations = last_obs

        next_values = self.agent.get_value(last_obs)  # selecting action

        returns, advs = compute_returns_advantages(
            rewards, dones, values, next_values, self.discount)  # getting advantage

        actions = stack_and_flatten_actions(all_actions)
        obs = flatten_first_dims_dict(stack_ndarray_dicts(all_observations))
        returns = flatten_first_dims(returns)
        advs = flatten_first_dims(advs)

        if self.train:
            return self.agent.train(
                obs, actions, returns, advs,
                summary=train_summary)

        return None

    def _summarize_episode(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        self.summary_writer.scalar_summary('sc2/episode_score', score, self.episode_counter)
        print("episode %d: score = %f" % (self.episode_counter, score))
        self.episode_counter += 1
        return score


def main():
    size_px = (args.res, args.res)
    env_args = dict(
        map_name=args.map,
        step_mul=args.step_mul,
        game_steps_per_episode=0,
        screen_size_px=size_px,
        minimap_size_px=size_px)

    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = args.vis

    num_vis = min(args.envs, args.max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = args.envs - num_vis
    if num_no_vis > 0:
        env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
    envs = SubprocVecEnv(env_fns)
    agent = A2CAgent(args)

    current_epoch = 0
    if os.path.isfile(args.save_dir + '.pth.tar') and not args.overwrite:
        current_epoch = agent.load_checkpoint()
        print("Restored from last checkpoint at epoch", current_epoch)

    summary_writer = Logger(args.summary_dir)

    runner = A2CRunner(
        envs=envs,
        agent=agent,
        train=args.train,
        summary_writer=summary_writer,
        discount=args.discount,
        n_steps=args.steps_per_batch)

    runner.reset()
    try:
        while True:
            if current_epoch % args.save_iters == 0:
                agent.save_checkpoint(current_epoch)
            result = runner.run_batch(train_summary=True)
            # agent.log(summary_writer, i)
            current_epoch += 1
    except KeyboardInterrupt:
        pass

    envs.close()
    print('mean score: %f' % runner.get_mean_score())


if __name__ == "__main__":
    main()
