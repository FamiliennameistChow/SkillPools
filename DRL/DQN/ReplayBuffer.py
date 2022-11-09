#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: DRL --- > ReplayBuffer.py.py
# Author: bornchow
# Time:20220522
# 使用np储存记忆
# ------------------------------------
import random

import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    def __init__(self, obs_dim, max_mem_size, batch_size):
        super(ReplayBuffer, self).__init__()
        self.ob_buf = np.zeros([max_mem_size, obs_dim], dtype=np.float32)
        self.next_ob_buf = np.zeros([max_mem_size, obs_dim], dtype=np.float32)
        self.action_buf = np.zeros([max_mem_size, 1], dtype=np.float32)
        self.reward_buf = np.zeros([max_mem_size, 1], dtype=np.float32)
        self.done_buf = np.zeros([max_mem_size, 1], dtype=bool)
        self.max_mem_size = max_mem_size
        self.batch_size = batch_size

        self.memory_cntr = 0
        self.memory_size = 0

    def store_transition(self, s, a, r, s_, done):
        index = self.memory_cntr % self.max_mem_size

        self.ob_buf[index] = s
        self.next_ob_buf[index] = s_
        self.action_buf[index] = a
        self.reward_buf[index] = r
        self.done_buf[index] = done

        self.memory_cntr += 1
        self.memory_size = min(self.memory_size + 1, self.max_mem_size)

    def sample_buffer(self):
        choice = np.random.choice(self.memory_size, self.batch_size)

        ob_buf = self.ob_buf[choice]
        next_ob_buf = self.next_ob_buf[choice]
        actions_buf = self.action_buf[choice]
        reward_buf = self.reward_buf[choice]
        done_buf = self.done_buf[choice]

        return ob_buf, actions_buf, reward_buf, next_ob_buf, done_buf

    def __len__(self):
        return self.memory_size

    def buffer_info(self):

        print("state size:", np.shape(self.ob_buf))
        print("action size:", np.shape(self.action_buf))
        print("reward size:", np.shape(self.reward_buf))


# https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, max_mem_size, batch_size, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, max_mem_size, batch_size)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        assert alpha >= 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_mem_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store_transition(self, s, a, r, s_, done):
        super().store_transition(s, a, r, s_, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_mem_size

    def sample_buffer(self, beta=0.4):
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs      = self.ob_buf[indices]
        next_obs = self.next_ob_buf[indices]
        actions  = self.action_buf[indices]
        reward   = self.reward_buf[indices]
        done     = self.done_buf[indices]

        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return obs, actions, reward, next_obs, done, weights, indices

    def update_priorities(self, indices, priorities):
        """
        update priorities of sampled transitions
        :param indices: List[int]
        :param priorities: np.ndarray
        :return: None
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _calculate_weight(self, idx, beta):
        """ calculate the weigth of the experience at idx"""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        return weight

    def _sample_proportional(self):
        """
        Sample indices based on proportions
        :return:  List[int]
        """
        indices = []
        p_total = self.sum_tree.sum(0, len(self)-1)

        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i+1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices


if __name__ == "__main__":
    import gym
    env = gym.make("CartPole-v1")
    env = env.unwrapped

    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]
    print(N_STATES)
    buffer = ReplayBuffer(N_STATES, 2000, 32)
    buffer.buffer_info()
