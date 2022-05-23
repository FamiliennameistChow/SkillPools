#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: DRL --- > run_dueling_dqn.py.py
# Author: bornchow
# Time:20220523
# DDQN + PER + dueling
#
# per部分用 用PER标注
# dueling 部分用dueling标注
# ------------------------------------
import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from ReplayBuffer import PrioritizedReplayBuffer
from torch.nn.utils import clip_grad_norm_


class NN(nn.Module):
    def __init__(self, state_dims, action_dims):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(state_dims, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        self.fc2.weight.data.normal_(0, 0.1)

        # advantage output
        self.adv = nn.Linear(128, action_dims)
        self.adv.weight.data.normal_(0, 0.1)
        # value output
        self.value = nn.Linear(128, 1)
        self.value.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # dueling dqn 主要修改部分
        value = self.value(x)  # [batch_size, 1]
        advantage = self.adv(x)  # [batch_size, action_dims]

        # 论文公式
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q


class DQN(object):
    def __init__(self,
                 action_dims,
                 states_dims,
                 max_memory_size=2000,
                 batch_size=32,
                 gamma=0.9,
                 lr=0.01,
                 min_epsilon=0.1,
                 max_epsilon=1.0,
                 epsilon_decay=1/1000,
                 target_update_iter=100,
                 # PER param
                 alpha=0.2,
                 beta=0.6,
                 prior_eps=1e-6
    ):
        self.action_dims = action_dims
        self.states_dims = states_dims
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon = max_epsilon
        self.target_update_iter = target_update_iter

        # eval_net and target_net
        self.eval_net = NN(self.states_dims, self.action_dims)  # 实时更新　Q估计　输入的是当前状态
        self.target_net = NN(self.states_dims, self.action_dims)  # 延时更新 Qtarget网络　输入的是下一个状态
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.lr)

        # PER
        # in dqn use self.memory = ReplayBuffer(self.states_dims, self.max_memory_size, self.batch_size)
        self.beta = beta
        self.alpha = alpha
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(self.states_dims, self.max_memory_size, self.batch_size, self.alpha)

        self.writer = SummaryWriter("./logs")

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < self.epsilon:  # 选取网络输出最大的值
            action = np.random.randint(0, N_ACTIONS)  # int
        else:  # 随机选取
            action_value = self.eval_net(x)
            action = torch.max(action_value, 1)[1].data.numpy()[0]

        return action

    def store_transition(self, s, a, r, s_):
        self.memory.store_transition(s, a, r, s_, done)

    def _target_hard_update(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        print("learn ......", self.beta, len(self.memory))

        # 每隔TARGET_REPLACE_ITER步更新　target_net　Q现实　参数
        if self.learn_step_counter % self.target_update_iter == 0:
            self._target_hard_update()

        # 每一步都更新 eval_net
        # 从记忆库里采样数据
        # PER need beta to cal weights
        np_s, np_a, np_r, np_s_, np_done, weights, indices = self.memory.sample_buffer(self.beta)
        b_s = torch.from_numpy(np_s).type(torch.FloatTensor)
        b_a = torch.from_numpy(np_a).type(torch.LongTensor)
        b_r = torch.from_numpy(np_r).type(torch.FloatTensor)
        b_s_ = torch.from_numpy(np_s_).type(torch.FloatTensor)
        b_done = torch.from_numpy(np_done).type(torch.FloatTensor)

        b_weights = torch.FloatTensor(weights).reshape(-1, 1)
        indices = indices

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape(batch_size, 1)
        # q_target   = r + gamma * v(s_{t+1})  if state != Terminal
        #            = r                           otherwise
        # ---DDQN的主要改动在 q_next 这里
        # in DQN
        # q_next = self.target_net(b_s_).max(1)[0].view(-1, 1).detach()  # shape (batch_size, 2)
        q_next = self.target_net(b_s_).gather(1, self.eval_net(b_s_).argmax(dim=1, keepdim=True)).detach()
        q_target = b_r + self.gamma * q_next * (1-b_done)  # shape(batch_size, 1)

        elementwise_loss = F.smooth_l1_loss(q_target, q_eval, reduction="none")

        # PER add weights to loss
        # importance sampling before average
        loss = torch.mean(elementwise_loss * b_weights)

        self.writer.add_scalar("loss", loss, self.learn_step_counter)

        # 训练
        self.optimizer.zero_grad()
        loss.backward()
        # dueling dqn: clip gradients to have their norm less than or equal to 10
        clip_grad_norm_(self.eval_net.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices=indices, priorities=new_priorities)

        # 更新贪婪系数
        self.epsilon = max(self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay)

        self.learn_step_counter += 1


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = env.unwrapped

    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]
    N_GAMES = 4000
    print(N_STATES)

    dqn = DQN(action_dims=N_ACTIONS, states_dims=N_STATES)
    print("collection experience.....")

    for i in range(N_GAMES):
        s = env.reset()
        episode_reward = 0

        # PER: increase beta
        # fraction = min(i / N_GAMES, 1.0)
        # dqn.beta = dqn.beta + fraction * (1.0 - dqn.beta)

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

            episode_reward += r

            dqn.store_transition(s, a, r, s_)

            if dqn.memory.memory_size > dqn.batch_size:
                dqn.learn()

            if done:
                break

            s = s_

        print("episode {}  reward {} ".format(i, episode_reward))
        dqn.writer.add_scalar("reward ", episode_reward, i)

    dqn.writer.close

    # sim_data = torch.rand((5, 4)).type(torch.FloatTensor)
    # out = dqn.eval_net(sim_data)
    #
    # print(out)
    #
    # print(out.max(1)[1])
    #
    # print(out.argmax(dim=1, keepdim=False))
    #
    # target_out = dqn.target_net(sim_data).gather(1, out.argmax(dim=1, keepdim=True))

