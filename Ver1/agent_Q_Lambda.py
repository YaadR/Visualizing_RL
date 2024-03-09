"""
Q(Lambda) Algorithm
"""

import numpy as np
from game import SnakeGameAI, Direction, Point
from helper import table_visualize, array_tobinary
import matplotlib.pyplot as plt
from settings import *

# Constants
S.EPSILON = 80  # Exploration rate


class Agent_Q_Lambda:
    def __init__(self):
        self.n_games = 0
        self.epsilon = S.EPSILON  # randomness
        self.gamma = S.GAMMA  # discount rate
        self.alpha = S.ALPHA
        self.Lambda = S.LAMBDA  # learning rate
        self.Q = dict()
        self.eligibility_trace = dict()
        self.num_actions = S.NUM_ACTIONS
        self.num_episodes = S.NUM_EPISODES
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
            game.food.x < game.head[id].x,  # food left
            game.food.x > game.head[id].x,  # food right
            game.food.y < game.head[id].y,  # food up
            game.food.y > game.head[id].y,  # food down
        ]

        return np.array(state, dtype=int)

    # Function to choose an action based on epsilon-greedy policy
    def get_action(self, state):
        uni = np.random.uniform()
        if uni < (self.epsilon - self.n_games) / self.num_episodes:
            # if self.n_games < self.epsilon:
            return np.random.randint(S.NUM_ACTIONS)
        else:
            state_idx = array_tobinary(state)
            self.actions_probability = self.Q[state_idx]
            return np.argmax(self.Q[state_idx])

    # Function to update Q-values using TD(0) learning
    def update_Q(self, state, action, reward, next_state):
        delta = (
            reward
            + self.gamma * np.max(self.Q[array_tobinary(next_state)])
            - self.Q[array_tobinary(state)][action]
        )
        self.eligibility_trace[array_tobinary(state)][action] += 1
        for key in self.Q.keys():
            for act in range(self.num_actions):
                self.Q[key][act] += (
                    self.alpha * delta * self.eligibility_trace[key][act]
                )
                self.eligibility_trace[key][act] *= self.gamma * self.Lambda

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        mean_score = 0
        record = 0
        # agent = AgentTD_Lambda()
        game = SnakeGameAI(arrow=True)

        plt.ion()

        fig, axs = plt.subplots(1, 2, width_ratios=[8, 4], figsize=(8, 6))
        while True:
            # get old state
            state = self.get_state(game)
            if array_tobinary(state) not in self.Q.keys():
                self.Q[array_tobinary(state)] = [0, 0, 0]
                self.eligibility_trace[array_tobinary(state)] = [0, 0, 0]

            # get move
            action = self.get_action(state)
            game.actions_probability = self.actions_probability

            # perform move and get new state
            reward, done, score = game.play_step(action)
            state_next = self.get_state(game)
            if array_tobinary(state_next) not in self.Q.keys():
                self.Q[array_tobinary(state_next)] = [0, 0, 0]
                self.eligibility_trace[array_tobinary(state_next)] = [0, 0, 0]

            self.update_Q(state, action, reward, state_next)

            if done:
                game.reset()
                self.n_games += 1

                if score > record:
                    record = score

                print("Game", self.n_games, "Score", score, "Record:", record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / self.n_games
                plot_mean_scores.append(mean_score)
                table_visualize(
                    np.array(list(self.Q.values())), axs, plot_mean_scores, plot_scores
                )

                if mean_score > 4:
                    break

    def play(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        game = SnakeGameAI(arrow=True, obstacle_flag=True)

        while True:
            # get old state
            state = self.get_state(game)
            if array_tobinary(state) not in self.Q.keys():
                self.Q[array_tobinary(state)] = [0, 0, 0]
                self.eligibility_trace[array_tobinary(state)] = [0, 0, 0]

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
                total_score += score
                mean_score = total_score / self.n_games
                plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)
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


def main():
    agent = Agent_Q_Lambda()
    agent.train()
    plt.close()
    agent.play()


if __name__ == "__main__":
    main()
