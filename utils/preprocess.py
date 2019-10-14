import numpy as np
from utils import arglist
from pysc2.lib import features

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index

def screen():
    screen_channel = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            screen_channel += 1
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            screen_channel += 1
        else:
            screen_channel += features.SCREEN_FEATURES[i].scale
    return screen_channel

def minimap():
    minimap_channel = 0
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID:
            minimap_channel += 1
        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            minimap_channel += 1
        else:
            minimap_channel += features.MINIMAP_FEATURES[i].scale
    return minimap_channel


def preprocess_screen(screen):
    layers = []
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE or features.SCREEN_FEATURES[
            i].type == features.FeatureType.SCALAR:
            layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
        else:
            layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
            for j in range(features.SCREEN_FEATURES[i].scale):
                y, x = (screen[i] == j).nonzero()
                layer[j, y, x] = 1
            layers.append(layer)
    return np.concatenate(layers, axis=0)


def preprocess_minimap(minimap):
    layers = []
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID or features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
        else:
            layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
            for j in range(features.MINIMAP_FEATURES[i].scale):
                y, x = (minimap[i] == j).nonzero()
                layer[j, y, x] = 1
            layers.append(layer)
    return np.concatenate(layers, axis=0)


class Preprocess:
    def __init__(self):
        self.num_screen_channels = len(features.SCREEN_FEATURES)
        self.num_minimap_channels = len(features.MINIMAP_FEATURES)
        self.num_flat_obs = arglist.NUM_ACTIONS
        self.available_actions_channels = arglist.NUM_ACTIONS

    def get_observation(self, state):
        obs_flat = state.observation['available_actions']
        obs_flat = self._onehot1d(obs_flat)

        obs = {'minimap': state.observation['feature_minimap'],
               'screen': state.observation['feature_screen'],
               'nonspatial': obs_flat}
        return obs

    def preprocess_action(self, act):
        return act

    def postprocess_action(self, act):
        '''
        input: action <FunctionCall>
        output: action <dict of np.array>
        '''
        act_categorical = np.zeros(shape=(arglist.NUM_ACTIONS,), dtype='float32')
        act_categorical[act.function] = 1.  # 0  0  1  0  0
        act_screens = [np.zeros(shape=(1, arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), dtype='float32')] * 2  # 2d projection
        i = 0
        for arg in act.arguments:
            if arg != [0]:
                act_screens[i][0, arg[0], arg[1]] = 1.
                i += 1

        act = {'categorical': act_categorical,
               'screen1': act_screens[0],
               'screen2': act_screens[1]}

        return act

    def _onehot1d(self, x):
        y = np.zeros((self.num_flat_obs,), dtype='float32')
        y[x] = 1.
        return y

