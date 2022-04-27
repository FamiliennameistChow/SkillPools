#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: DRL  --- > buffer.py
# Author: bornchow
# Date:20210623
# Link:https://www.bilibili.com/video/BV1ha411A7pR/?spm_id_from=333.788.recommend_more_video.-1
# ------------------------------------
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.new_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, n_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, s, a, r, s_, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.new_state_memory[index] = s_
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory_counter = min(self.memory_counter, self.memory_size)

        sample_index = np.random.choice(max_memory_counter, batch_size)

        sample_states = self.state_memory[sample_index]
        sample_states_ = self.new_state_memory[sample_index]
        sample_actions = self.action_memory[sample_index]
        sample_rewards = self.reward_memory[sample_index]
        sample_dones = self.terminal_memory[sample_index]

        return sample_states, sample_actions, sample_rewards, sample_states_, sample_dones


if __name__ == "__main__":
    buffer = ReplayBuffer(5000, [8], 2)


