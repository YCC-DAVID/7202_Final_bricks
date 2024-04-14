import gym
import numpy as np
import matplotlib.pyplot as plt
from RL_brain import PolicyGradient
import torch

# ------------------------------- #
# 模型参数设置
# ------------------------------- #

n_hiddens = 512  # 隐含层个数
learning_rate = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
return_list = []  # 保存每回合的reward
max_q_value = 0  # 初始的动作价值函数
max_q_value_list = []  # 保存每一step的动作价值函数

# ------------------------------- #
# （1）加载环境
# ------------------------------- #

# 连续性动作
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 4
n_actions = env.action_space.n  # 动作数 2

# ------------------------------- #
# （2）模型实例化
# ------------------------------- #

agent = PolicyGradient(n_states=n_states,  # 4
                       n_hiddens=n_hiddens,  # 16
                       n_actions=n_actions,  # 2
                       learning_rate=learning_rate,  # 学习率
                       gamma=gamma)  # 折扣因子

# ------------------------------- #
# （3）训练
# ------------------------------- #

for i in range(150):  # 训练10回合
    # 记录每个回合的return
    episode_return = 0
    # 存放状态
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }
    # 获取初始状态
    state = env.reset()[0]
    # 结束的标记
    done = False
    max_q_value_one_turn = 0
    n_action = 0

    # 开始迭代
    while not done:
        reward_o = 0
        # 动作选择
        action = agent.take_action(state)  # 对某一状态采取动作
        # 动作价值函数，曲线平滑
        max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
        # 保存每一step的动作价值函数
        max_q_value_one_turn += max_q_value
        n_action += 1
        # max_q_value_list.append(max_q_value)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        reward_o += reward
        # modify the reward
        x, x_dot, theta, theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2


        # 保存每个回合的所有信息
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 状态更新
        state = next_state
        # 记录每个回合的return
        episode_return += reward_o

    max_q_value_list.append(max_q_value_one_turn/n_action)
    # 保存每个回合的return
    return_list.append(episode_return)
    # 一整个回合走完了再训练模型
    agent.learn(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# 关闭动画
env.close()
model_path = './policy_gradient_model.pth'
torch.save(agent.policy_net.state_dict(), model_path)
# -------------------------------------- #
# 绘图
# -------------------------------------- #

import pandas as pd

# 假设 return_list 已经定义并包含数据
# 创建一个 DataFrame 来存储这些数据
df_returns = pd.DataFrame(return_list, columns=['Returns'])

# 保存 DataFrame 到 CSV 文件
csv_returns_path = 'returns_data.csv'
df_returns.to_csv(csv_returns_path, index=False)

print(f"Returns data saved to {csv_returns_path}")
# 假设 max_q_value_list 已经定义并包含数据
# 创建一个 DataFrame 来存储这些数据
df_max_q_values = pd.DataFrame(max_q_value_list, columns=['MaxQValues'])

# 保存 DataFrame 到 CSV 文件
csv_max_q_values_path = 'max_q_values_data.csv'
df_max_q_values.to_csv(csv_max_q_values_path, index=False)

print(f"Max Q Values data saved to {csv_max_q_values_path}")



def test_model(model_path, env, agent, num_episodes=100):
    # 加载模型
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()  # 设置为评估模式

    for i in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_return = 0
        while not done:
            action = agent.take_action(state, use_random=False)  # 使用确定性策略
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_return += reward
        print(f"Episode {i + 1}: Return = {episode_return}")


# test_model(model_path, env, agent, num_episodes=10)
