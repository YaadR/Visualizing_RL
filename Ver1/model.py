import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import warnings


# Enable displaying warnings as regular text messages
warnings.filterwarnings("ignore")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x=self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

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

class Dyna_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)


    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# class ActorCritic(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)
#         self.fc2 = nn.Linear(256, output_size)
#         self.fc3 = nn.Linear(256, 1)
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#
#         # Check the shape of the tensor
#         if len(x.shape) == 1:
#             # If the tensor has only one dimension, apply softmax along dimension 0
#             action_probs = F.softmax(self.fc2(x), dim=0)
#         else:
#             # If the tensor has more than one dimension, apply softmax along dimension 1
#             action_probs = F.softmax(self.fc2(x), dim=1)
#
#         state_value = self.fc3(x)
#         return action_probs, state_value

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


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size,hidden_size=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        probs = torch.softmax(self.fc2(x), dim=1)
        return probs

class CriticNetwork(nn.Module):
    def __init__(self, state_size,hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size) # Added 1 for action size
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value


class _DPN_Trainer_:
    def __init__(self, critic_network,actor_network, lr, gamma,critic_optimizer,actor_optimizer):
        self.lr = lr
        self.gamma = gamma
        self.critic_network = critic_network
        self.actor_network = actor_network
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.criterion = None
        self.loss_bus = 0


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


        # Calculate the TD-error
        value = self.critic_network(state)
        next_value = self.critic_network(next_state)
        td_error = reward + self.gamma * (1 - done) * next_value - value

        # Update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss = torch.mean(td_error.pow(2))
        # if loss_bus_flag:
            # print(critic_loss)
        # self.loss_bus = critic_loss.detach().numpy()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        action_probs = self.actor_network(state)
        log_probs = torch.log(action_probs.squeeze(0)[action])
        actor_loss = -torch.matmul(log_probs.T, td_error.detach())

        actor_loss = torch.mean(actor_loss) #+ critic_loss.detach()
        # if loss_bus_flag:
            # print(actor_loss)
        # self.loss_bus = actor_loss.detach().numpy()
        # actor_loss.backward()
        loss = actor_loss + critic_loss
        loss.backward()
        self.loss_bus = loss.detach().numpy()
        self.actor_optimizer.step()
        self.critic_optimizer.step()



class DPN_Trainer:
    def __init__(self, model, lr, gamma, optimizer,scheduler = None):
        self.lr = lr
        self.gamma = gamma
        self.model = model
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

        _, values = self.model(state)
        _, next_values = self.model(next_state)

        td_targets = reward + self.gamma * next_values * (1 - done)
        advantages = td_targets - values

        # Actor loss
        log_probs, _ = self.model(state)
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
        action_probs, state_value = self.model(state)

        # Compute the TD error
        next_action_probs, next_state_value = self.model(next_state)
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


class ValueTrainer:
    def __init__(self, model, lr, gamma,alpha=1):
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_bus = 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # done = (done, )
            done = int(done)


        # predicted Q values with current state
        # pred = self.model(state)
        V = self.model(state)
        V_next = self.model(next_state)


        # target = pred.clone()
        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #
        #     target[idx][torch.argmax(action[idx]).item()] = Q_new

        state_value = V.clone()
        next_state_value = V_next.clone()
        # V(s) = V(s) + alpha *(reward +gamma *V(s')-V(s))

        value = state_value + self.alpha * (reward[0]+(1-done)*(self.gamma*next_state_value - state_value))

        implicit_action = torch.argmax(action[0]).item()
        state_value[0][implicit_action] = value[0][implicit_action].item() #  if not done[0] else value
        # state_value[idx] = value
        self.optimizer.zero_grad()
        loss = self.criterion(state_value, V)
        loss.backward()
        self.optimizer.step()

class Value_Trainer:
    def __init__(self, model, lr, gamma,alpha=1):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.alpha = alpha
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
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
        state_value = self.model(state)
        next_state_value = self.model(*next_state)


        target = state_value.clone()
        for idx in range(len(done)):
            # V(s') = V(s)+alpha*(reward + (1-terminal_state)*(gamma*next_state - sate)
            state_value_prime = state_value + self.alpha * (reward[0][idx] + (1-done[idx])*(self.gamma * next_state_value[idx] - state_value))
            target[0][idx] = max(state_value_prime[0])


        self.optimizer.zero_grad()
        loss = self.criterion(target, state_value)
        loss.backward()
        self.optimizer.step()

class Value_Trainer_1:
    def __init__(self, model, lr, gamma,alpha=1):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.alpha = alpha
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
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
        state_value = self.model(state)
        next_state_value = self.model(*next_state)


        target = state_value.clone()
        for idx in range(len(done)):
            state_value_prime = state_value + self.alpha * (reward[0][idx] + (1-done[idx])*(self.gamma * next_state_value[idx] - state_value))
            target[0][0] = max(state_value_prime[0])

        self.optimizer.zero_grad()
        loss = self.criterion(target, state_value)
        loss.backward()
        self.optimizer.step()


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
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
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        if loss_bus_flag:
            self.loss_bus = loss.detach().numpy()
        loss.backward()
        self.optimizer.step()


