import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import time
import random
from urllib.request import proxy_bypass
from math import pi







def visualize_weights(model):
    # Extract the weights from the model
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy())

    # Plot the weights
    fig, axs = plt.subplots(len(weights), 1, figsize=(8, 6))
    for i, weight in enumerate(weights):
        if len(weight.shape) > 1:
            axs[i].imshow(weight.squeeze(), cmap='gray')
        else:
            axs[i].plot(weight)
        axs[i].set_title(f'Layer {i + 1} weights')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.pause(10)


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate random data for 10 lists of mean scores
# num_lists = 10
# num_scores = 100
# mean_scores = []
# scores = []
# for _ in range(num_lists):
#     scores.append(list(np.random.rand(num_scores)))
#
#
# for n in range(num_lists):
#     for i in range(100):
#         scores[n][i]*=5*i
#
#
# mean_scores =  np.mean(scores,axis=0)
#
# # Plotting parameters
# alpha_value = 0.2  # Alpha value for faded background
# average_color = 'blue'  # Color for the average line
#
# # Plot each list of mean scores with a faded background
# for i, score in enumerate(scores):
#     plt.plot(score, color='orange', alpha=alpha_value)
#
# # Plot the average line with a solid color
# # average_scores = np.mean(mean_scores, axis=0)
# plt.plot(mean_scores, color=average_color, label='Average')
#
# # Add labels and title
# plt.xlabel('Iterations')
# plt.ylabel('Mean Score')
# plt.title('Average of Multiple Lists of Mean Scores')
#
# # Add legend
# plt.legend()
#
# # Show the plot
# plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

# PPO Algorithm
def ppo_algorithm(env, num_episodes, max_timesteps, batch_size, epsilon, hidden_dim, gamma, lr, clip_ratio, num_epochs):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = []

        for t in range(max_timesteps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            episode_rewards.append(reward)

            if t == max_timesteps - 1:
                done = True

            state = next_state

            if done or t == max_timesteps - 1:
                # Calculate cumulative returns (discounted rewards)
                cumulative_returns = np.zeros_like(episode_rewards)
                discounted_reward = 0
                for i in reversed(range(len(episode_rewards))):
                    discounted_reward = gamma * discounted_reward + episode_rewards[i]
                    cumulative_returns[i] = discounted_reward

                # Normalize cumulative returns
                cumulative_returns = (cumulative_returns - cumulative_returns.mean()) / (cumulative_returns.std() + 1e-8)

                # Convert to PyTorch tensors
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)
                returns_tensor = torch.FloatTensor(cumulative_returns)

                # PPO policy update
                for epoch in range(num_epochs):
                    action_probs = policy_net(states_tensor)
                    action_dist = torch.distributions.Categorical(action_probs)
                    log_prob = action_dist.log_prob(actions_tensor)

                    # Calculate advantage estimates
                    advantage = returns_tensor - values  # You'll need to compute the values tensor

                    # Calculate surrogate objective
                    ratio = torch.exp(log_prob - old_log_probs)
                    surrogate1 = ratio * advantage
                    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
                    surrogate_obj = torch.min(surrogate1, surrogate2)

                    # Clip the surrogate objective
                    clipped_surrogate_obj = -torch.mean(surrogate_obj)

                    # Update the policy network using gradient ascent
                    optimizer.zero_grad()
                    clipped_surrogate_obj.backward()
                    optimizer.step()

    env.close()

# Example usage
import gym

env = gym.make('CartPole-v1')
num_episodes = 100
max_timesteps = 200
batch_size = 32
epsilon = 0.2
hidden_dim = 64
gamma = 0.99
lr = 0.001
clip_ratio = 0.2
num_epochs = 10

ppo_algorithm(env, num_episodes, max_timesteps, batch_size, epsilon, hidden_dim, gamma, lr, clip_ratio, num_epochs)
