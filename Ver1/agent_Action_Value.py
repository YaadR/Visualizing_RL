"""
Action Value Agent:
 - Model free
 - off policy
  - online
 - value based : action value
"""

from settings import S
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, pygame
from model import Linear_Net, Value_Trainer_A
from helper import heat_map_step, distance_collapse, activation_visualize
from helper import array_tobinary, entropy, softmax, cirtenty_function

from path import BASE_DIR

import matplotlib.pyplot as plt


class Action_Value:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.net = Linear_Net(S.STATE_VEC_SIZE, S.HIDDEN_LAYER, S.NUM_ACTIONS)
        self.trainer = Value_Trainer_A(self.net, lr=S.LR, gamma=self.gamma)
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
            game.food.y > game.head[id].y  # food down
            # Food distance from head - X axis, Y axis and both
            # preprocessing.normalize([[math.dist([game.head[id].x], [game.food.x]), 0, game.w]])[0][0],
            # preprocessing.normalize([[math.dist([game.head[id].y], [game.food.y]), 0, game.h]])[0][0]
        ]

        return np.array(state, dtype=int)

    def train_online(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            self.prediction = action
            action[move] = 1
        else:
            prediction = self.net(torch.tensor(state, dtype=torch.float))
            self.prediction = softmax(prediction.detach().numpy())
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        game = SnakeGameAI(arrow=False, agentID=0, certainty_flag=True)
        mean_score = 0
        seen_states = set()
        counter = 0
        screen_sample = 20
        system_entropy = []
        mean_entropy = deque(maxlen=10)  # Moving average of decision entropy

        heatmap = np.ones((game.w // 10, game.h // 10))  # Heatmap init
        plt.ion()

        heat_flag = False
        layers_flag = False
        if heat_flag:
            figure, axis = plt.subplots(1, 2, width_ratios=[2, 3], figsize=(10, 4))
            axis[0].set_title("Heatmap")

        while True:
            # get old state
            state_prev = self.get_state(game)

            # get move
            action = self.get_action(state_prev)
            game.actions_probability = self.prediction
            mean_entropy.append(cirtenty_function(entropy(self.prediction)))
            game.certainty = np.round(np.mean(mean_entropy), 5)
            # print(game.certainty)

            # perform move and get new state
            reward, done, score = game.play_step(action)
            state = self.get_state(game)

            # update heatmap
            if heat_flag:
                if reward:
                    # reset heatmap
                    heatmap[:] = 1
                else:
                    X, Y = distance_collapse(
                        state[-2:], game.w, game.h, game.direction.value
                    )
                    heatmap = heat_map_step(
                        heatmap,
                        game.direction.value,
                        game.w // 10,
                        game.h // 10,
                        int(game.head.x) // 10,
                        int(game.head.y) // 10,
                        any(state[:3]),
                        X // 10,
                        Y // 10,
                    )

            if (
                (mean_score > 12) or (self.n_games <= 15 and self.n_games >= 5)
            ) and heat_flag:
                axis[0].imshow(heatmap.T, cmap="viridis", interpolation="nearest")

            if self.n_games == 20 or self.n_games == 80 or self.n_games == 150:
                if not counter % 50:
                    pygame.image.save(
                        game.display,
                        BASE_DIR / f"Ver1/data/plots/Certainty/certain_{counter}.jpg",
                    )
                counter += 1

            # train short memory
            self.train_online(state_prev, action, reward, state, done)

            # Activation layer of every unique state
            if layers_flag and mean_score > 15 and len(game.snake) > 20:
                if array_tobinary(state) not in seen_states:
                    plt.close()
                    plt.ion()
                    fig, axs = plt.subplots(
                        1, 4, width_ratios=[1, 3, 1, 5], figsize=(10, 6)
                    )
                    seen_states.add(array_tobinary(state))
                    env = pygame.surfarray.array3d(game.display)
                    layer_1_activation = self.net.linear1(
                        torch.tensor(state, dtype=torch.float)
                    )
                    layer_2_activation = (
                        self.net.linear2(torch.relu(layer_1_activation))
                        .detach()
                        .numpy()
                    )
                    layer_1_activation = layer_1_activation.detach().numpy()
                    activation_visualize(
                        state.reshape((1, -1)),
                        layer_1_activation,
                        layer_2_activation.reshape((1, -1)),
                        axs,
                        env,
                        array_tobinary(state),
                    )

            if done:
                # plot result
                game.reset()
                self.n_games += 1
                if heat_flag:
                    heatmap[:] = 1

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

                if heat_flag:
                    axis[1].cla()
                    axis[1].set_title("Training")
                    axis[1].set_xlabel("Games")
                    axis[1].set_ylabel("Score")
                    axis[1].plot(plot_scores)
                    axis[1].plot(plot_mean_scores)
                    axis[1].set_ylim(ymin=0)
                    axis[1].text(
                        len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1])
                    )
                    axis[1].text(
                        len(plot_mean_scores) - 1,
                        plot_mean_scores[-1],
                        str(round(plot_mean_scores[-1], 3)),
                    )

            if S.USE_HEAT_MAP:
                if (
                    ((mean_score > 12) or (self.n_games <= 15 and self.n_games >= 5))
                    and heat_flag
                ) or done:
                    plt.show()
                    plt.pause(0.001)

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
    agent = Action_Value()
    agent.train()
    plt.close()
    agent.play()
