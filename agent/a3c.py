import os
import numpy as np
import tensorflow as tf
import random
from pysc2.lib import actions, features
import utils
from agent.agent import Agent


UPDATE_GLOBAL_INTERVAL = 5
GAMMA = 0.99
MAX_EPISODE = 20000
N_HIDDEN = 200
MAX_EPISODE_STEP = 200


class A3CAgent(Agent):
    def __init(self, actor, critic, discount, entropy_weight, value_loss_weight, resolution, training,  exploration_mode):
        self.name = 'a3c'
        self.discount = discount
        self.entropy_weight = entropy_weight
        self.value_loss_weight = value_loss_weight
        self.training = training
        self.summary = []
        self.resolution = resolution
        self.structured_dimensions = len(actions.FUNCTIONS)
        self.mode = exploration_mode

    def reset(self)
        self.epsilon = [0.05, 0.2]

