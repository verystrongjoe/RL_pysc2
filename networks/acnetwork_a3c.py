'''
Actor and critic networks share conv. layers to process minimap & screen.
'''

from math import floor
import torch
import torch.nn as nn
from utils import arglist
from utils.layers import TimeDistributed, Flatten, Dense2Conv, init_weights


minimap_channels = 7
screen_channels = 17
flat_observation = 11  #  changed.

# apply paddinga as 'same', padding = (kernel - 1)/2
conv_minimap = nn.Sequential(nn.Conv2d(minimap_channels, 16, 5, stride=1, padding=2),  # shape (N, 16, m, m)
                             nn.ReLU(),
                             nn.Conv2d(16, 32, 3, stride=1, padding=1),  # shape (N, 32, m, m)
                             nn.ReLU())

conv_screen = nn.Sequential(nn.Conv2d(screen_channels, 16, 5, stride=1, padding=2),  # shape (N, 16, m, m)
                            nn.ReLU(),
                            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # shape (N, 32, m, m)
                            nn.ReLU())

dense_nonspatial = nn.Sequential(nn.Linear(flat_observation, 32),
                                 nn.ReLU(),
                                 Dense2Conv())

class A3CNet(torch.nn.Module):
    def __init__(self):
        super(A3CNet, self).__init__()
        # spatial features
        self.minimap_conv_layers = conv_minimap
        self.screen_conv_layers = conv_screen
        self.nonspatial_dense = dense_nonspatial # non-spatial features

        # state representations
        self.layer_hidden = nn.Sequential(nn.Conv2d(32 * 3, 64, 3, stride=1, padding=1),
                                          nn.ReLU())
        # output layers
        self.layer_action = nn.Sequential(nn.Conv2d(64, 1, 1),
                                          nn.ReLU(),
                                          Flatten(),
                                          nn.Linear(arglist.FEAT2DSIZE * arglist.FEAT2DSIZE, arglist.NUM_ACTIONS))
        self.layer_screen1 = nn.Conv2d(64, 1, 1)
        self.layer_screen2 = nn.Conv2d(64, 1, 1)

        self.apply(init_weights)  # weight initialization
        self.train()  # train mode

    def forward(self, obs):
        obs_minimap = obs['minimap']
        obs_screen = obs['screen']
        obs_nonspatial = obs['nonspatial']
        
        # process observations
        m = self.minimap_conv_layers(obs_minimap)
        s = self.screen_conv_layers(obs_screen)
        n = self.nonspatial_dense(obs_nonspatial)

        state_h = torch.cat([m, s, n], dim=1)
        state_h = self.layer_hidden(state_h)

        pol_categorical = self.layer_action(state_h)
        pol_screen1 = self.layer_screen1(state_h)
        pol_screen2 = self.layer_screen2(state_h)

        v = self.layer_value(state_h)

        return {'categorical': pol_categorical, 'screen1': pol_screen1, 'screen2': pol_screen2, "value": v }


