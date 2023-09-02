import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import warnings


# Enable displaying warnings as regular text messages
warnings.filterwarnings("ignore")


class Linear_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class Linear_Net_Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        value = self.linear2(x)
        policy = F.softmax(value)
        return policy, value


# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_actor = nn.Linear(64, output_size)
        self.fc_critic = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # policy = torch.log(F.softmax(self.fc_actor(x), dim=-1))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value


class A2C_Trainer:
    def __init__(self, net, lr, gamma, optimizer):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.optimizer = optimizer
        self.critic_criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor([done], dtype=torch.int)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)
            # done =  (done, )

        _, values = self.net(state)
        _, next_values = self.net(next_state)

        advantages = reward + (1 - done) * (self.gamma * next_values - values)

        # Actor loss
        probs, _ = self.net(state)
        log_probs = torch.log(probs)
        log_probs = log_probs.gather(1, action.unsqueeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = self.critic_criterion(values, advantages.detach())
        loss = actor_loss + critic_loss
        # Update the network
        self.optimizer.zero_grad()
        # actor_loss.backward()
        # critic_loss.backward()
        loss.backward()
        self.optimizer.step()


# Model based trainer state value
class Value_Trainer_V:
    def __init__(self, net, lr, gamma, alpha=1):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.alpha = alpha
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_bus = 0

    def train_step(self, state, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            # done = (done, )

        # predicted values with current state
        state_value = self.net(state)
        next_state_value = self.net(*next_state)

        target = state_value.clone()
        for idx in range(len(done)):
            # V(s') = V(s)+alpha*(reward + (1-terminal_state)*(gamma*next_state - sate)
            state_value_prime = state_value + self.alpha * (
                reward[0][idx]
                + (1 - done[idx]) * (self.gamma * next_state_value[idx] - state_value)
            )
            target[0][idx] = max(state_value_prime[0])

        self.optimizer.zero_grad()
        loss = self.criterion(target, state_value)
        loss.backward()
        self.optimizer.step()


# Model free trainer state value
class State_Value_Trainer:
    def __init__(self, net, lr, gamma, alpha=1):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.alpha = alpha
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_bus = 0

    def train_step(self, state, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            # done = (done, )
            done = int(done)

        # predicted values with current state
        state_value = self.net(state)
        next_state_value = self.net(next_state)

        target = state_value.clone()
        state_value_prime = state_value + self.alpha * (
            reward[0] + (1 - done) * (self.gamma * next_state_value - state_value)
        )
        target[0][0] = state_value_prime[0]

        self.optimizer.zero_grad()
        loss = self.criterion(target, state_value)
        loss.backward()
        self.optimizer.step()


# Model free trainer action value
class Value_Trainer_A:
    def __init__(self, net, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_bus = 0

    def train_step(self, state_prev, action, reward, state, done):
        state_prev = torch.tensor(state_prev, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        loss_bus_flag = False

        state_prev = torch.unsqueeze(state_prev, 0)
        state = torch.unsqueeze(state, 0)
        action = torch.unsqueeze(action, 0)
        reward = torch.unsqueeze(reward, 0)

        # predicted Q values with current state
        pred = self.net(state_prev)

        target = pred.clone()
        # Q' = Reward + (1-terminal)*lmbda*max(Actions_Value)
        Q_new = reward[0] + (1 - done) * self.gamma * torch.max(self.net(state[0]))
        target[0][torch.argmax(action[0]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        if loss_bus_flag:
            self.loss_bus = loss.detach().numpy()
        loss.backward()
        self.optimizer.step()
