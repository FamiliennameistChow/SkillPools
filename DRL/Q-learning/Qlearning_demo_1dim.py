# -*- coding: UTF-8 -*-
import numpy as np 
import pandas as pd 
import time

np.random.seed(2) 

N_STATES = 6  # 一维世界的长度
ACTION = ["left", "right"] #可能的动作
EPSILON = 0.9 # greedy police
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 # 奖励的衰减值
MAX_EPISODES = 13 # max episodes
FRESH_TIME = 0.3 # fresh time for one move


def build_q_table(n_states, action):
    table = pd.DataFrame(
        np.zeros((n_states, len(action))),
        columns=action,
    )
    # print(table)
    return table


def choose_action(state, q_table):
    # this is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all()==0):
        action_name = np.random.choice(ACTION)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    # this is how agent will interact with the environment
    if A == "right":
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # this is how env be update
    env_list = ["-"]*(N_STATES-1) + ["T"]
    if S == "terminal":
        interaction = "Episode %s: total_setps = %s" % (episode+1, step_counter)
        print("\r{}".format(interaction), end=" ")
        time.sleep(2)
        print("\r                               ", end=" ")
    else:
        env_list[S] = "o"
        interaction = ''.join(env_list)
        print("\r{}".format(interaction), end="")
        time.sleep(FRESH_TIME)
    # print(env_list)


def rl():
    # main part of loop
    q_table = build_q_table(N_STATES, ACTION)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)

        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA*q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            
            q_table.loc[S, A] += ALPHA*(q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter += 1
    
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nq-table:\n')
    print(q_table)







