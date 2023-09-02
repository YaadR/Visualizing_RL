"""
Agent Value:
 - Model free
 - off policy
  - online
 - value based : state value
"""

import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point, pygame
from model import Linear_Net, State_Value_Trainer, nn
from helper import plot, array_tobinary
import matplotlib.pyplot as plt
from settings import *

S.ALPHA = 0.3  # Learning rate


class Agent_State_Value:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = S.GAMMA  # discount rate
        self.alpha = S.ALPHA  #
        self.Q = dict()  # Q table
        self.net = Linear_Net(S.STATE_VEC_SIZE, S.HIDDEN_LAYER, S.STATE_VALUE)
        self.trainer = State_Value_Trainer(
            self.net, lr=S.LR, gamma=self.gamma, alpha=self.alpha
        )
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
            game.food.y > game.head.y  # food down
            # Food distance from head - X axis, Y axis and both
            # preprocessing.normalize([[math.dist([game.head.x],[game.food.x]),0,game.w]])[0][0],
            # preprocessing.normalize([[math.dist([game.head.y],[game.food.y]),0,game.h]])[0][0]
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

    def train_online(self, state, reward, next_state, done):
        self.trainer.train_step(state, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            self.actions_probability = action
            action[move] = 1
        else:
            prediction = self.Q[array_tobinary(state)]
            self.actions_probability = prediction
            move = np.argmax(prediction)
            action[move] = 1

        return action

    def update_Q(self, s, a, s_prime):
        self.Q[array_tobinary(s)][np.argmax(a)] = (
            self.net(torch.tensor(s_prime, dtype=torch.float)).detach().numpy()[0]
        )

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        game = SnakeGameAI(arrow=True, agentID=0)
        mean_score = 0

        plt.ion()

        while True:
            # get old state
            state_prev = self.get_state(game)

            if array_tobinary(state_prev) not in self.Q.keys():
                self.Q[array_tobinary(state_prev)] = [0, 0, 0]
            # get move

            action = self.get_action(state_prev)
            game.actions_probability = self.actions_probability

            reward, done, score = game.play_step(action)
            state = self.get_state(game)

            # train value approximation
            self.train_online(state_prev, reward, state, done)
            self.update_Q(state_prev, action, state)

            if done:
                # plot result
                game.reset()
                self.n_games += 1

                if score > record:
                    record = score

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / self.n_games
                print(
                    "Game:",
                    self.n_games,
                    "Score:",
                    score,
                    "Record:",
                    record,
                    "Mean Score:",
                    round(mean_score, 3),
                )
                plot_mean_scores.append(mean_score)

                plot(plot_scores, plot_mean_scores)
                if mean_score > 12:
                    break

    def play(self):
        plot_scores = []
        record = 0
        game = SnakeGameAI(arrow=True, obstacle_flag=True)

        while True:
            # get old state
            state = self.get_state(game)
            # get move
            action = self.get_action(state)
            game.actions_probability = self.actions_probability
            # perform move and get new state
            reward, done, score = game.play_step(action)
            if done:
                game.reset()
                self.n_games += 1

                if score > record:
                    record = score
                plot_scores.append(score)

                # plot(plot_scores, plot_mean_scores)
                print(
                    "Game:",
                    self.n_games,
                    "Score:",
                    score,
                    "Record:",
                    record,
                    "Mean Score:",
                )


def main():
    agent = Agent_State_Value()

    agent.train()
    agent.play()


if __name__ == "__main__":
    main()
