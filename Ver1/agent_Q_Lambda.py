'''
Q(Lambda) Algorithm
'''

import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, pygame
from helper import plot,heat_map_step,distance_collapse,table_visualize,array_tobinary
import matplotlib.pyplot as plt
import json
import statistics

# Constants
NUM_ACTIONS = 3  # Number of possible actions (up, down, left, right)
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 80 # Exploration rate
NUM_EPISODES = 100  # Number of training episodes

LAMBDA = 0.8

BLOCK_SIZE = 20
WIDTH =  480
HEIGHT = 360
STATE_VEC_SIZE =11
MAX_MEMORY = 100_000

class Agent_Q_Lambda:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON # randomness
        self.gamma = GAMMA # discount rate
        self.alpha = ALPHA
        self.Lambda = LAMBDA # learning rate
        # self.Q = np.zeros((2**STATE_VEC_SIZE, NUM_ACTIONS)) # Initialize Q-table
        self.Q = dict()
        self.eligibility_trace  = dict()
        self.num_actions = NUM_ACTIONS
        self.num_episodes = NUM_EPISODES
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
        # min_env = np.zeros((HEIGHT // BLOCK_SIZE, WIDTH // BLOCK_SIZE))
        # for x, i in enumerate(range(0, HEIGHT - BLOCK_SIZE, BLOCK_SIZE)):
        #     for y, j in enumerate(range(0, WIDTH - BLOCK_SIZE, BLOCK_SIZE)):
        #         min_env[x, y] = np.sum(np.sum(env[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE], axis=2)) // BLOCK_SIZE ** 2
        # state  = [Danger straight,Danger right,Danger left,L,R,U,D,food left,food right,food up,food down]

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
        ]
        return np.array(state, dtype=int)

    def get_state_arena(self, game,id=0):
        head = game.snake[id][0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction[id].value == Direction.LEFT.value
        dir_r = game.direction[id].value == Direction.RIGHT.value
        dir_u = game.direction[id].value == Direction.UP.value
        dir_d = game.direction[id].value == Direction.DOWN.value

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
            game.food.x < game.head[id].x,  # food left
            game.food.x > game.head[id].x,  # food right
            game.food.y < game.head[id].y,  # food up
            game.food.y > game.head[id].y,  # food down

        ]

        return np.array(state, dtype=int)

    # Function to choose an action based on epsilon-greedy policy
    def get_action(self,state):
        uni = np.random.uniform()
        if uni < (self.epsilon-self.n_games)/self.num_episodes:
        # if self.n_games < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        else:
            state_idx = array_tobinary(state)
            self.actions_probability = self.Q[state_idx]
            return np.argmax(self.Q[state_idx])


    # Function to update Q-values using TD(0) learning
    def update_Q(self,state, action, reward, next_state):

        delta = reward + self.gamma * np.max(self.Q[array_tobinary(next_state)]) - self.Q[array_tobinary(state)][action]
        self.eligibility_trace[array_tobinary(state)][action] += 1
        for key in self.Q.keys():
            for act in range(self.num_actions):
                self.Q[key][act] += self.alpha * delta * self.eligibility_trace[key][act]
                self.eligibility_trace[key][act] *= self.gamma * self.Lambda

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    mean_score = 0
    record = 0
    # agent = AgentTD_Lambda()
    game = SnakeGameAI(arrow=True)

    plt.ion()

    fig, axs = plt.subplots(1, 2,width_ratios=[8,4], figsize=(8, 6))
    while True:
        # get old state
        state = agent.get_state(game)
        if array_tobinary(state) not in agent.Q.keys():
            agent.Q[array_tobinary(state)] = [0,0,0]
            agent.eligibility_trace[array_tobinary(state)] = [0,0,0]

        # get move
        action = agent.get_action(state)
        game.actions_probability = agent.actions_probability

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_next = agent.get_state(game)
        if array_tobinary(state_next) not in agent.Q.keys():
            agent.Q[array_tobinary(state_next)] = [0,0,0]
            agent.eligibility_trace[array_tobinary(state_next)] = [0, 0, 0]


        agent.update_Q(state,action,reward,state_next)


        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score


            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            table_visualize(np.array(list(agent.Q.values())), axs,plot_mean_scores,plot_scores)
            # plot(plot_scores, plot_mean_scores)
            if mean_score>4:
                break

def play():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI(arrow=True,obstacle_flag=True)

    while True:
        # get old state
        state = agent.get_state(game)
        if array_tobinary(state) not in agent.Q.keys():
            agent.Q[array_tobinary(state)] = [0,0,0]
            agent.eligibility_trace[array_tobinary(state)] = [0,0,0]

        # get move
        action = agent.get_action(state)
        game.actions_probability = agent.actions_probability

        # perform move and get new state
        reward, done, score = game.play_step(action)
        # state_next = agent.get_state(game)
        # if array_tobinary(state_next) not in agent.Q.keys():
        #     agent.Q[array_tobinary(state_next)] = [0,0,0]
        #     agent.eligibility_trace[array_tobinary(state_next)] = [0, 0, 0]

        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            print('Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Mean Score:', round(mean_score,3))

if __name__ == '__main__':
    agent = Agent_Q_Lambda()
    train()
    play()