import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        x=self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class Linear_Net_Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x=self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x)
        return x


# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self,  input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_actor = nn.Linear(64, output_size)
        self.fc_critic = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = torch.log(F.softmax(self.fc_actor(x), dim=-1))
        value = self.fc_critic(x)
        return policy, value


class A2C_Trainer:
    def __init__(self, net, lr, gamma, optimizer,scheduler = None):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.optimizer = optimizer
        self.criterion = None
        self.loss_actor = 0
        self.loss_critic = 0
        self.scheduler = scheduler

    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor([done], dtype=torch.int)
        # (n, x)
        loss_bus_flag = False

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)
            #done =  (done, )
        else:
            loss_bus_flag = True

        _, values = self.net(state)
        _, next_values = self.net(next_state)

        td_targets = reward + self.gamma * next_values * (1 - done)
        advantages = td_targets - values

        # Actor loss
        log_probs, _ = self.net(state)
        log_probs = log_probs.gather(1, action.unsqueeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = F.mse_loss(values, td_targets.detach())

        # Update the network
        self.optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer.step()
        self.loss_critic = critic_loss.detach().numpy()
        self.loss_actor = actor_loss.detach().numpy()

    def train_step_(self, state, action, reward, next_state, done):
        # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        # action = torch.tensor([action], dtype=torch.long)
        # reward = torch.tensor([reward], dtype=torch.float)
        # next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor([done], dtype=torch.int)
        # (n, x)
        loss_bus_flag = False

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = torch.unsqueeze(done, 0)
            #done =  (done, )
        else:
            loss_bus_flag = True


        # Compute the predicted action probabilities and state value
        action_probs, state_value = self.net(state)

        # Compute the TD error
        next_action_probs, next_state_value = self.net(next_state)
        td_error = reward + self.gamma * next_state_value * (1 - done) - state_value

        # Compute the actor and critic losses
        log_action_probs = torch.log(action_probs)
        actor_loss = -log_action_probs[0][action] * td_error
        critic_loss = F.smooth_l1_loss(state_value, reward + self.gamma * next_state_value * (1 - done))

        # Update the networks
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()


# Model based trainer state value
class Value_Trainer_V:
    def __init__(self, net, lr, gamma,alpha=1):
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
            state_value_prime = state_value + self.alpha * (reward[0][idx] + (1-done[idx])*(self.gamma * next_state_value[idx] - state_value))
            target[0][idx] = max(state_value_prime[0])


        self.optimizer.zero_grad()
        loss = self.criterion(target, state_value)
        loss.backward()
        self.optimizer.step()

# Model free trainer state value
class Value_Trainer_1:
    def __init__(self, net, lr, gamma,alpha=1):
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
            state_value_prime = state_value + self.alpha * (reward[0][idx] + (1-done[idx])*(self.gamma * next_state_value[idx] - state_value))
            target[0][0] = max(state_value_prime[0])

        self.optimizer.zero_grad()
        loss = self.criterion(target, state_value)
        loss.backward()
        self.optimizer.step()

# Model free trainer on policy
class Policy_Trainer_A:
    def __init__(self, net, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.net = net
        self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_bus = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        loss_bus_flag = False
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        else:
            loss_bus_flag = True

        # predicted Q values with current state
        pred = self.net(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.net(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        if loss_bus_flag:
            self.loss_bus = loss.detach().numpy()
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

    def train_step(self, state_old, action, reward, state, done):
        state_old = torch.tensor(state_old, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        loss_bus_flag = False
        if len(state_old.shape) == 1:
            # (1, x)
            state_old = torch.unsqueeze(state_old, 0)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        else:
            loss_bus_flag = True

        # predicted Q values with current state
        pred = self.net(state_old)

        target = pred.clone()
        for idx in range(len(done)):
            # Q' = Reward : Terminal state
            Q_new = reward[idx]
            if not done[idx]:
                # Q' = Reward + lmbda*max(Actions_Value)
                Q_new = reward[idx] + self.gamma * torch.max(self.net(state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        if loss_bus_flag:
            self.loss_bus = loss.detach().numpy()
        loss.backward()
        self.optimizer.step()


