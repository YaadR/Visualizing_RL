'''
Agent Value:
 - Model based
 - off policy
 - online
 - value based : state value
'''

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, pygame
from model import Linear_Net, Value_Trainer_V
from helper import plot, heat_map_step, distance_collapse, net_visualize, activation_visualize,normalizer,plot_mean_scores_buffer,plot_std_mean_scores_buffer
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import statistics


ALPHA = 0.5  # Learning rate
GAMMA = 0.9  # Discount factor
##
BLOCK_SIZE = 20
WIDTH = 480
HEIGHT = 360
##

LR = 0.001
NUM_ACTIONS = 3  # Number of possible actions (up, down, left, right)
STATE_VEC_SIZE = 11
HIDDEN_LAYER = 256


class Agent_Value:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = GAMMA  # discount rate
        self.alpha = ALPHA #
        self.net = Linear_Net(STATE_VEC_SIZE, HIDDEN_LAYER, NUM_ACTIONS)
        self.trainer = Value_Trainer_V(self.net, lr=LR, gamma=self.gamma,alpha=self.alpha)
        self.prediction = [0, 0, 0]
        self.env_model = SnakeGameAI()

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
            game.food.y > game.head.y  # food down

            # Food distance from head - X axis, Y axis and both
            # preprocessing.normalize([[math.dist([game.head.x],[game.food.x]),0,game.w]])[0][0],
            # preprocessing.normalize([[math.dist([game.head.y],[game.food.y]),0,game.h]])[0][0]
        ]

        return np.array(state, dtype=float)

    def get_state_arena(self, game, id=0):
        head = game.snake[id][0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction[id].value == Direction.LEFT.value
        dir_r = game.direction[id].value == Direction.RIGHT.value
        dir_u = game.direction[id].value == Direction.UP.value
        dir_d = game.direction[id].value == Direction.DOWN.value

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

            # Food distance from head - X axis, Y axis and both
            # preprocessing.normalize([[math.dist([game.head[id].x], [game.food.x]), 0, game.w]])[0][0],
            # preprocessing.normalize([[math.dist([game.head[id].y], [game.food.y]), 0, game.h]])[0][0]
        ]

        return np.array(state, dtype=float)

    def train_online(self, state, reward, next_state, done):
        self.trainer.train_step(state, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 10 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            self.states_value = action
            action[move] = 1
        else:
            self.prediction = self.net(torch.tensor(state, dtype=torch.float)).detach().numpy()
            move = np.argmax(self.prediction)
            action[move] = 1

        return action

    def get_states_value(self,game):
        # [Straight, Right, Left]
        rewards, dones,next_states = [],[],[]
        actions = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
        for i,action in enumerate(actions):
            self.env_model = game.copy(self.env_model)
            reward, done, score = self.env_model.play_step(action)
            next_state = self.get_state(self.env_model)
            rewards.append(reward), dones.append(done), next_states.append(next_state)

        dones = np.array(dones).astype(int)
        return rewards, dones,next_states


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI(arrow=True, agentID=0)
    mean_score = 0

    plt.ion()

    # fig, axs = plt.subplots(1, 3,width_ratios=[4,1,6], figsize=(8, 6))
    # fig, axs = plt.subplots(1, 4, width_ratios=[12, 4, 8, 1], figsize=(8, 6))

    while True:
        # get old state
        state = agent.get_state(game)

        # get move
        action = agent.get_action(state)
        game.actions_probability = agent.prediction

        # perform move and get new state
        _reward, _done, _state_next = agent.get_states_value(game)

        reward, done, score = game.play_step(action)

        # train short memory
        agent.train_online(state, _reward, _state_next, _done)

        if done:
            # plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            print('Games:', i,'Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Mean Score:', round(mean_score, 3))
            plot_mean_scores.append(mean_score)

            # plot(plot_scores, plot_mean_scores)

            if agent.n_games>=80:
                mean_scores.append(list(plot_mean_scores))
                break


def play():
    plot_scores = []
    record = 0
    game = SnakeGameAI(arrow=True, obstacle_flag=True)

    while True:
        # get old state
        state = agent.get_state(game)
        # get move
        action = agent.get_action(state)
        # perform move and get new state
        reward, done, score = game.play_step(action)
        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
            plot_scores.append(score)

            # plot(plot_scores, plot_mean_scores)
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Mean Score:')



if __name__ == '__main__':
    agent = Agent_Value()
    mean_scores = []
    i=0

    train()
    plt.close()
    play()

    # for i in range(20):
    #     agent = Agent_Value()
    #     train()
    # name = 'State Value Agents'
    # plot_mean_scores_buffer(mean_scores,name)
    # plot_std_mean_scores_buffer(mean_scores,name)




