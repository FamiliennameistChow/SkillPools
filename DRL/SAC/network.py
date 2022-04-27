#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: DRL --- > SAC_Network.py
# Author: bornchow
# Time:20210622
# Link: https://www.bilibili.com/video/BV1ha411A7pR/?spm_id_from=333.788.recommend_more_video.-1
# ------------------------------------
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal


class CriticNet(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name="critic", fc1_dims=256, fc2_dims=256,  chkpt_dir="./model"):
        super(CriticNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+"_sac.pth")

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):  # state shape 1*n
        x = self.fc1(torch.cat([state, action], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        q = self.q(x)
        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class ValueNet(nn.Module):
    def __init__(self, beta, input_dims, name="value", fc1_dims=256, fc2_dims=256,  chkpt_dir="./model"):
        super(ValueNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+"_sac.pth")
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class ActorNet(nn.Module):
    def __init__(self, alpha, input_dims, max_action,
                 fc1_dims=256, fc2_dims=256, n_actions=2, name="actor", chkpt_dir="./model"):
        super(ActorNet, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name+"_sac.pth")
        self.max_action = max_action
        self.n_actions = n_actions
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        prob = Normal(mu, sigma)  # 创建由mu和sigma参数化的正态分布

        if reparameterize:
            actions = prob.rsample()
        else:
            actions = prob.sample()  # normally distributed with loc=mu and scale=sigma
        # 来源于论文
        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = prob.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdims=True)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


if __name__ == "__main__":
    # critic net test
    critic_net = CriticNet(beta=0.001, input_dims=[3], n_actions=1, name="critic")
    print(critic_net)

    state = torch.tensor([3, 1, 1.], device=critic_net.device).view(1, -1)   # 注意输入要是TensorFloat32
    action = torch.tensor([2]).view(1, -1).to(critic_net.device)
    print("state: ", state.shape)
    print("action: ", action.shape)

    c = torch.cat([state, action], dim=1)
    print("critic input ", c.shape)

    a = critic_net(state, action)
    print(a.shape)

    # print(dict(critic_net.named_parameters()))

    print(critic_net.state_dict())

    for name in critic_net.state_dict():
        print(name)
