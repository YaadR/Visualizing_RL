'''
Deep Policy Network, also known as a Policy Gradient Network
'''

#
import torch
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import ActorCritic, ActorNetwork, CriticNetwork, DPN_Trainer, DPN_Trainer
from helper import plot, visualize_biases,net_visualize
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import statistics
import warnings
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore")

##
BLOCK_SIZE = 20
WIDTH = 480
HEIGHT = 360
##

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
NUM_ACTIONS = 3  # Number of possible actions (up, down, left, right)
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
NUM_EPISODES = 100  # Number of training episodes
STATE_VEC_SIZE = 11
HIDDEN_LAYER = 256


class AgentDPN:

    def __init__(self):
        self.n_games = 0
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        # Actor Critic combined
        self.model = ActorCritic(STATE_VEC_SIZE, NUM_ACTIONS)  # Linear_QNet(13, 256, 3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        # self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.trainer = DPN_Trainer(model=self.model, optimizer=self.optimizer, lr=LR, gamma=self.gamma)

        # Actor Critic apart
        # self.critic_network = CriticNetwork(STATE_VEC_SIZE,hidden_size=HIDDEN_LAYER)
        # self.actor_network = ActorNetwork(STATE_VEC_SIZE,NUM_ACTIONS,hidden_size=HIDDEN_LAYER)
        # self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=LR)
        # self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=LR)
        # self.trainer = DPN_Trainer(critic_network= self.critic_network,actor_network=self.actor_network, critic_optimizer=self.critic_optimizer,actor_optimizer=self.actor_optimizer, lr=LR, gamma=self.gamma)


        self.actions_probability = [0, 0, 0]

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # env = pygame.surfarray.array3d(game.display)
        # min_env = np.zeros((HEIGHT//BLOCK_SIZE,WIDTH//BLOCK_SIZE))
        # for x,i in enumerate(range(0,HEIGHT-BLOCK_SIZE,BLOCK_SIZE)):
        #     for y,j in enumerate(range(0,WIDTH-BLOCK_SIZE,BLOCK_SIZE)):
        #         min_env[x, y] = np.sum(np.sum(env[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE],axis=2))//BLOCK_SIZE**2

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Food distance from head - X axis, Y axis and both
            #round(preprocessing.normalize([[math.dist([game.head.x], [game.food.x]), 0, game.w]])[0][0],2),
            #round(preprocessing.normalize([[math.dist([game.head.y], [game.food.y]), 0, game.h]])[0][0],2)

        ]

        return np.array(state, dtype=float)

    def get_state_arena(self, game, id=0):
        head = game.snake[id][0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction[id] == Direction.LEFT
        dir_r = game.direction[id] == Direction.RIGHT
        dir_u = game.direction[id] == Direction.UP
        dir_d = game.direction[id] == Direction.DOWN

        # env = pygame.surfarray.array3d(game.display)
        # min_env = np.zeros((HEIGHT//BLOCK_SIZE,WIDTH//BLOCK_SIZE))
        # for x,i in enumerate(range(0,HEIGHT-BLOCK_SIZE,BLOCK_SIZE)):
        #     for y,j in enumerate(range(0,WIDTH-BLOCK_SIZE,BLOCK_SIZE)):
        #         min_env[x, y] = np.sum(np.sum(env[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE],axis=2))//BLOCK_SIZE**2

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r, id)) or
            (dir_l and game.is_collision(point_l, id)) or
            (dir_u and game.is_collision(point_u, id)) or
            (dir_d and game.is_collision(point_d, id)),

            # Danger right
            (dir_u and game.is_collision(point_r, id)) or
            (dir_d and game.is_collision(point_l, id)) or
            (dir_l and game.is_collision(point_u, id)) or
            (dir_r and game.is_collision(point_d, id)),

            # Danger left
            (dir_d and game.is_collision(point_r, id)) or
            (dir_u and game.is_collision(point_l, id)) or
            (dir_r and game.is_collision(point_u, id)) or
            (dir_l and game.is_collision(point_d, id)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head[id].x,  # food left
            game.food.x > game.head[id].x,  # food right
            game.food.y < game.head[id].y,  # food up
            game.food.y > game.head[id].y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        self.actions_probability = [0, 0, 0]

        action_probs,_ = self.model(state)
        action = action_probs[0].squeeze().detach().numpy()
        self.actions_probability = action

        action_probs = action_probs.squeeze().detach().numpy()
        action = np.random.choice(NUM_ACTIONS, p=action_probs)
        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    mean_score = 0
    agent = AgentDPN()
    game = SnakeGameAI(arrow=True, agentID=0)
    loss_buss = []
    loss_buss_2 = []
    last_bias = np.zeros((int(np.sqrt(HIDDEN_LAYER)), int(np.sqrt(HIDDEN_LAYER))))
    difference_val = []
    episode_loss = []
    episode_loss_2 = []
    distance_collapse = 2

    plt.ion()

    # fig, axs = plt.subplots(1, 4, width_ratios=[4, 1,1, 6], figsize=(8, 6))
    # fig, axs = plt.subplots(1, 5, width_ratios=[16, 8, 12,1], figsize=(8, 6))
    fig, axs = plt.subplots(1, 5, width_ratios=[4,4, 1, 1,7], figsize=(8, 6))
    # fig, axs = plt.subplots(1, 8, width_ratios=[16, 8, 12,1], figsize=(8, 6))
    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        action = agent.get_action(state_old)
        game.actions_probability = agent.actions_probability

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        #### Altering the reward function
        # if not done and False:
        #     # if np.any(state_new[:3]) and not reward: # warning over danger ahead
        #     #     reward = -1
        #     if distance_collapse > 2*np.sum(state_new[-2:]) and not reward: # Distance collapse towards food
        #         distance_collapse = np.sum(state_new[-2:])
        #         reward = 0.1
        #     elif reward > 0.1: # reset food distance after food was reached
        #         distance_collapse = 2



        # train short memory
        agent.train_short_memory(state_old, np.argmax(action), reward, state_new, int(done))
        episode_loss.append(agent.trainer.loss_critic)
        episode_loss_2.append(agent.trainer.loss_actor)

        # remember
        agent.remember(state_old, np.argmax(action), reward, state_new, int(done))

        if done:
            # train long memory, plot result
            game.reset()
            distance_collapse = 2
            agent.n_games += 1
            agent.train_long_memory()
            loss_buss.append(np.mean(episode_loss))
            loss_buss_2.append(np.mean(episode_loss_2))



            if score > record:
                record = score

            # print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            last_bias = visualize_biases(agent.model, axs, last_bias, difference_val, loss_buss,epsilon_decay=plot_mean_scores, agent_type=1,loss_1=loss_buss_2)
            # net_visualize(agent.model, axs)
            # difference_val[0] = 0
            episode_loss = []
            episode_loss_2 = []
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Mean Score:',round(mean_score, 3))


if __name__ == '__main__':
    train()
