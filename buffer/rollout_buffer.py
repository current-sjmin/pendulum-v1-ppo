import numpy as np


class RolloutBuffer(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self, state, action, reward, done, value, log_prob):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.dones.append(np.array(done, dtype=np.float32))
        self.values.append(np.array(value, dtype=np.float32))
        self.log_probs.append(np.array(log_prob, dtype=np.float32))

    def get(self):
        return {
            "states"    : self.states,
            "actions"   : self.actions,
            "rewards"   : self.rewards,
            "dones"     : self.dones,
            "values"    : self.values,
            "log_probs" : self.log_probs
        }

    def clear(self):
        self.__init__()