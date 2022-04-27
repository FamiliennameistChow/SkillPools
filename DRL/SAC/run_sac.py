#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: DRL  --- > SAC_model.py
# Author: bornchow
# Date:20210623
# Link: https://www.bilibili.com/video/BV1ha411A7pR/?spm_id_from=333.788.recommend_more_video.-1
# ------------------------------------

import os
import torch
import torch.nn.functional as F
from buffer import ReplayBuffer
from network import ActorNet, CriticNet, ValueNet

import pybullet_envs
import gym
import numpy as np
from utils import plot_learning_curve
from torch.utils.tensorboard import SummaryWriter


class Agent(object):
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None, gamma=0.99, n_actions=2,
                 max_size=100000, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNet(alpha, input_dims, n_actions=n_actions, name="actor", max_action=env.action_space.high)

        self.critic_1 = CriticNet(beta, input_dims, n_actions=n_actions, name="critic_1")
        self.critic_2 = CriticNet(beta, input_dims, n_actions=n_actions, name="critic_2")

        self.value = ValueNet(beta, input_dims, name="value")
        self.target_value = ValueNet(beta, input_dims, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

        self.writer = SummaryWriter("./logs")
        self.learn_step_counter = 0

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32).to(self.actor.device)

        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def add_memory(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()  # Q
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)  # target value的更新方式
        value_state_dict = dict(value_params)

        # 　更新value网络参数　tau是权重
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_model(self):
        print("......save models......")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_model(self):
        print(".....load models.......")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        # 　将数据转到tensor格式
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)
        state_ = torch.tensor(state_, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)

        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # uodate value Net
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        if self.learn_step_counter == 0:
            # self.writer.add_graph(self.value, state)
            self.writer.add_graph(self.target_value, state_)

        # update actor Net
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        #  update critic Net
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # update target_value Net
        self.update_network_parameters()
        self.learn_step_counter += 1


if __name__ == "__main__":
    env = gym.make("InvertedPendulumBulletEnv-v0")
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    n_games = 250
    filename = "inverted_pendulum.png"
    figure_file = "./plots/" + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_model()
        env.render(mode="human")

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            agent.add_memory(observation, action, reward, observation_, done)

            if not load_checkpoint:
                agent.learn()

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_model()

        print("episode %d | score %.1f avg_score %.1f" % (i, score, avg_score))

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    agent.writer.close()