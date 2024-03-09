"""
Policy Gradient Network | A2C Network

 - Model free
 - on policy
 - online
 - value based : state value
 - policy based

"""

from settings import *
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import ActorCritic, A2C_Trainer

import matplotlib.pyplot as plt
import warnings


mean_scores = []
i = 0


class Agent_Policy:
    def __init__(self):
        self.n_games = 0
        self.gamma = S.GAMMA  # discount rate
        self.memory = deque(maxlen=S.MAX_MEMORY)  # popleft()
        self.epsilon = 0

        # Actor Critic combined
        self.net = ActorCritic(
            S.STATE_VEC_SIZE, S.NUM_ACTIONS
        )  # Linear_QNet(11, 256, 3)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=S.LR)
        self.trainer = A2C_Trainer(
            net=self.net, optimizer=self.optimizer, lr=S.LR, gamma=self.gamma
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
            move = np.random.choice(S.NUM_ACTIONS, p=self.prediction)
            action[move] = 1
        else:
            prediction, _ = self.net(torch.tensor(state, dtype=torch.float))
            self.prediction = prediction.detach().numpy()
            move = np.argmax(self.prediction)
            action[move] = 1

        return action

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        mean_score = 0
        game = SnakeGameAI(arrow=True, agentID=0)
        plt.ion()

        while True:
            # get old state
            state_old = self.get_state(game)
            # get move
            action = self.get_action(state_old)
            game.actions_probability = self.prediction

            # perform move and get new state
            reward, done, score = game.play_step(action)
            state_new = self.get_state(game)

            # train short memory
            self.train_online(
                state_old, np.argmax(action), reward, state_new, int(done)
            )

            if done:
                # plot result
                game.reset()
                self.n_games += 1

                if score > record:
                    record = score

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / self.n_games
                plot_mean_scores.append(mean_score)
                if self.n_games >= 400:
                    mean_scores.append(list(plot_mean_scores))
                    break

                # plot(plot_scores, plot_mean_scores)
                print(
                    "Games:",
                    i,
                    "Game:",
                    self.n_games,
                    "Score:",
                    score,
                    "Record:",
                    record,
                    "Mean Score:",
                    round(mean_score, 3),
                )

    def play(self):
        plot_scores = []
        record = 0
        game = SnakeGameAI(arrow=True, obstacle_flag=True)

        while True:
            # get old state
            state = self.get_state(game)
            # get move
            action = self.get_action(state)
            game.actions_probability = self.prediction
            # perform move and get new state
            reward, done, score = game.play_step(action)
            if done:
                game.reset()
                self.n_games += 1

                if score > record:
                    record = score
                plot_scores.append(score)


def main():
    warnings.filterwarnings("ignore")
    agent = Agent_Policy()
    agent.train()
    plt.close()
    agent.play()


if __name__ == "__main__":
    main()
