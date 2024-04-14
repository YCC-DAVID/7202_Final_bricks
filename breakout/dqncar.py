import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0',render_mode="human")
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
return_list = []  # 保存每回合的reward
max_q_value = 0  # 初始的动作价值函数
max_q_value_list = []  # 保存每一step的动作价值函数

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(100):
    s, info = env.reset()
    ep_r = 0
    r_o = 0
    max_q_value_one_turn = 0
    n_action = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done,truncated, info = env.step(a)
        r_o += r
        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        current_s = torch.FloatTensor([s_])
        current_a = torch.LongTensor([a])
        current_q_values = dqn.eval_net(current_s)
        current_q_value = current_q_values.gather(1, current_a.unsqueeze(1))
        max_q_value_one_turn += max_q_value
        n_action += 1

        ep_r += r
        if i_episode > 32:
            dqn.learn()

            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
    return_list.append(r_o)
    max_q_value_list.append(max_q_value_one_turn / n_action)
import pandas as pd

# 假设 return_list 已经定义并包含数据
# 创建一个 DataFrame 来存储这些数据
df_returns = pd.DataFrame(return_list, columns=['Returns'])

# 保存 DataFrame 到 CSV 文件
csv_returns_path = 'returns_cartpole_dqn.csv'
df_returns.to_csv(csv_returns_path, index=False)

print(f"Returns data saved to {csv_returns_path}")
# 假设 max_q_value_list 已经定义并包含数据
# 创建一个 DataFrame 来存储这些数据
df_max_q_values = pd.DataFrame(max_q_value_list, columns=['MaxQValues'])

# 保存 DataFrame 到 CSV 文件
csv_max_q_values_path = 'max_q_values_cartpole_dqn.csv'
df_max_q_values.to_csv(csv_max_q_values_path, index=False)

print(f"Max Q Values data saved to {csv_max_q_values_path}")
