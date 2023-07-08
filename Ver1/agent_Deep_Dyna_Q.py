'''
 Deep Dyna-Q algorithm
'''

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, pygame
from model import Dyna_QNet, QTrainer
from helper import plot, heat_map_step, distance_collapse
from sklearn import preprocessing
import math
import pygame
import matplotlib.pyplot as plt
import statistics

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
EPSILON = 80 # Exploration rate
NUM_EPISODES = 10  # Number of training episodes
STATE_VEC_SIZE =13


class Agent_Deep_Dyna_Q:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Dyna_QNet(STATE_VEC_SIZE+NUM_ACTIONS, 256, NUM_ACTIONS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
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
            preprocessing.normalize([[math.dist([game.head.x], [game.food.x]), 0, game.w]])[0][0],
            preprocessing.normalize([[math.dist([game.head.y], [game.food.y]), 0, game.h]])[0][0]
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
            game.food.y > game.head[id].y,  # food down

            # Food distance from head - X axis, Y axis and both
            preprocessing.normalize([[math.dist([game.head[id].x], [game.food.x]), 0, game.w]])[0][0],
            preprocessing.normalize([[math.dist([game.head[id].y], [game.food.y]), 0, game.h]])[0][0]
        ]

        return np.array(state, dtype=float)

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

    def get_action(self, state,action):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        action_next = [0, 0, 0]
        #if random.randint(0, 200) < self.epsilon:
        if 0 < self.epsilon:
            move = random.randint(0, 2)
            self.actions_probability = action_next
            action_next[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0,action)
            self.actions_probability = prediction.detach().numpy()
            move = torch.argmax(prediction).item()
            action_next[move] = 1

        return action_next


def train():
    plot_scores = []
    plot_mean_scores = []
    record = 0
    agent = Agent_Deep_Dyna_Q()
    game = SnakeGameAI(arrow=True, agentID=0)
    ma_50 = deque(maxlen=50)
    action = [0,0,0]


    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        action = agent.get_action(state_old,action)
        game.actions_probability = agent.actions_probability

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)


        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score


            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            ma_50.append(score)
            plot_mean_scores.append(statistics.mean(ma_50))
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
