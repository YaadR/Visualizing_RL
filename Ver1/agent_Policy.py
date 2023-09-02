"""
Policy Gradient Network | A2C Network

 - Model free
 - on policy
 - online
 - value based : state value
 - policy based

"""

#
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import ActorCritic, A2C_Trainer

import matplotlib.pyplot as plt
import warnings

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
EPSILON = 50
NUM_EPISODES = 100  # Number of training episodes
STATE_VEC_SIZE = 11
HIDDEN_LAYER = 256


class Agent_Policy:
    def __init__(self):
        self.n_games = 0
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.epsilon = 0

        # Actor Critic combined
        self.net = ActorCritic(STATE_VEC_SIZE, NUM_ACTIONS)  # Linear_QNet(11, 256, 3)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.trainer = A2C_Trainer(
            net=self.net, optimizer=self.optimizer, lr=LR, gamma=self.gamma
        )
        self.prediction = [0, 0, 0]

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

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
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
            # round(preprocessing.normalize([[math.dist([game.head.x], [game.food.x]), 0, game.w]])[0][0],2),
            # round(preprocessing.normalize([[math.dist([game.head.y], [game.food.y]), 0, game.h]])[0][0],2)
        ]

        return np.array(state, dtype=int)

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
            (dir_r and game.is_collision(point_r, id))
            or (dir_l and game.is_collision(point_l, id))
            or (dir_u and game.is_collision(point_u, id))
            or (dir_d and game.is_collision(point_d, id)),
            # Danger right
            (dir_u and game.is_collision(point_r, id))
            or (dir_d and game.is_collision(point_l, id))
            or (dir_l and game.is_collision(point_u, id))
            or (dir_r and game.is_collision(point_d, id)),
            # Danger left
            (dir_d and game.is_collision(point_r, id))
            or (dir_u and game.is_collision(point_l, id))
            or (dir_r and game.is_collision(point_u, id))
            or (dir_l and game.is_collision(point_d, id)),
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

    def train_online(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # policy based exploration / exploitation
        self.epsilon = 50 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            prediction, _ = self.net(torch.tensor(state, dtype=torch.float))
            self.prediction = prediction.squeeze().detach().numpy()
            move = np.random.choice(NUM_ACTIONS, p=self.prediction)
            action[move] = 1
        else:
            prediction, _ = self.net(torch.tensor(state, dtype=torch.float))
            self.prediction = prediction.detach().numpy()
            move = np.argmax(self.prediction)
            action[move] = 1

        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    mean_score = 0
    game = SnakeGameAI(arrow=True, agentID=0)
    plt.ion()

    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        action = agent.get_action(state_old)
        game.actions_probability = agent.prediction

        # perform move and get new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_online(state_old, np.argmax(action), reward, state_new, int(done))

        if done:
            # plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if agent.n_games >= 400:
                mean_scores.append(list(plot_mean_scores))
                break

            # plot(plot_scores, plot_mean_scores)
            print(
                "Games:",
                i,
                "Game:",
                agent.n_games,
                "Score:",
                score,
                "Record:",
                record,
                "Mean Score:",
                round(mean_score, 3),
            )


def play():
    plot_scores = []
    record = 0
    game = SnakeGameAI(arrow=True, obstacle_flag=True)

    while True:
        # get old state
        state = agent.get_state(game)
        # get move
        action = agent.get_action(state)
        game.actions_probability = agent.prediction
        # perform move and get new state
        reward, done, score = game.play_step(action)
        if done:
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score
            plot_scores.append(score)

            # plot(plot_scores, plot_mean_scores)
            # print('Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Mean Score:')


if __name__ == "__main__":
    agent = Agent_Policy()
    mean_scores = []
    i = 0

    # for i in range(20):
    #     agent = Agent_Policy()
    #     train()
    # name = 'Action Policy Agents'
    # plot_mean_scores_buffer(mean_scores,name)
    # plot_std_mean_scores_buffer(mean_scores,name)

    train()
    plt.close()
    play()
