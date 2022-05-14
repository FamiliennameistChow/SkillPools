#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: DRL --- > run_dqn.py.py
# Author: bornchow
# Time:20220515
#
# ------------------------------------
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make("CartPole-v0")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

print(N_STATES)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(10, N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


class DQN(object):
    def __init__(self):
        self.eval_net = NN()  # 实时更新　Q估计　输入的是当前状态
        self.target_net = NN()  # 延时更新 Qtarget网络　输入的是下一个状态
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON: # 选取网络输出最大的值
            action_value = self.eval_net(x)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else: # 随机选取
            action = np.random.randint(0,N_ACTIONS) # int

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 每隔TARGET_REPLACE_ITER步更新　target_net　Q现实　参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # 每一步都更新 eval_net
        # 从记忆库里采样数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape(batch_size, 1)
        q_next = self.target_net(b_s_).detach()  # shape (batch_size, 2)
        q_target = b_r + GAMMA*q_next.max(1)[0].view(-1, 1)  # shape(batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        # 训练
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1


dqn = DQN()

print("collection experience.....")

writer = SummaryWriter("logs")

for i in range(400):
    s = env.reset()

    while True:
        env.render()
        # 观测
        a = dqn.choose_action(s)

        # take action互动
        s_, r, done, info = env.step(a)

        # 修改 reward, 使 DQN 快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            print("learn {}".format(dqn.learn_step_counter))
            writer.add_scalar("reward ", r, dqn.learn_step_counter)

        if done:
            break

        s = s_

writer.close()
