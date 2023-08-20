import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import copy
from pygame.locals import *

pygame.init()
# font = pygame.font.Font('./Ver1/arial.ttf', 25)
font = pygame.font.SysFont('arial', 18,bold=True)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

BLUE_FOOD = (0, 0, 150)
GREEN = (0, 100, 0)
GREEN_FOOD = (0, 150, 0)
PURPLE = (128, 0, 128)
PURPLE_FOOD = (224, 0, 224)
YELLOW = (128, 0, 128)

BLOCK_SIZE = 20
WIDTH = 480
HEIGHT = 360
SPEED = 60

OBSTACLE_HEIGHT = 60
OBSTACLE_WIDTH = 80
BROWN = (139, 69, 19)

AGENT_UI = [[BLUE, RED], [BLUE, BLUE_FOOD], [PURPLE, PURPLE_FOOD], [GREEN, GREEN_FOOD]]

text_position = [[0, 0],[WIDTH//2 -50, 0],[WIDTH-160, 0]]
AGENT_NAMES = ["Action Value", "Policy","State Value"]

class SnakeGameAI:

    def __init__(self, w=WIDTH, h=HEIGHT, arrow=False, agentID=0,obstacle_flag=False):
        self.w = w
        self.h = h
        self.arrow = arrow
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Environment')
        self.clock = pygame.time.Clock()
        self.obst_flag = obstacle_flag
        self.reset()
        self.actions_probability = [0, 0, 0]
        self.probability_clock = [0, 0, 0, 0]
        self.agentID = agentID

    def copy(self,copied_game):
        # copied_game = SnakeGameAI()
        excluded_attributes = ["clock"]  # Add any other attribute names you want to exclude
        for name, attr in self.__dict__.items():
            if name not in excluded_attributes:
                if hasattr(attr, 'copy') and callable(getattr(attr, 'copy')):
                    copied_game.__dict__[name] = attr.copy()
                else:
                    copied_game.__dict__[name] = copy.deepcopy(attr)
        return copied_game

    def place_obstacle(self):
        # self.obstacle = [Point(80, 60), Point(80, 240), Point(320, 60), Point(320, 240)]
        for W, H in [(80, 60), (80, 240), (320, 60), (320, 240)]:
            for h in range(H, H + OBSTACLE_HEIGHT, BLOCK_SIZE):
                for w in range(W, W + OBSTACLE_HEIGHT, BLOCK_SIZE):
                    self.obstacle.append(Point(w, h))

    def reset(self):
        # init game state
        self.obstacle = []
        if self.obst_flag:
            self.place_obstacle()   ###
        self.direction = Direction.RIGHT
        self.actions_probability = [0, 0, 0]
        self.probability_clock = [0, 0, 0, 0]

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacle:
            self._place_food()

    def play_step(self, action, get_env=False):
        self.frame_iteration += 1


        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 50 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()

        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacle:
            return True

        return False

    def is_food_ahead(self, pt=None):
        if pt is None:
            pt = self.head
        # hits itself
        if pt in self.food:
            return True
        return False

    def arrow_ui(self):
        if np.any(self.probability_clock):
            self.probability_clock = (self.probability_clock - np.min(self.probability_clock)) / (
                        np.max(self.probability_clock) - np.min(self.probability_clock))
        self.probability_clock *= 20
        correction = BLOCK_SIZE // 2
        ahead = 30
        ratio = 3

        if self.direction == Direction.RIGHT or Direction.UP or Direction.DOWN:
            arrow_points = [(self.head.x + ahead,
                             self.head.y - self.probability_clock[Direction.RIGHT.value - 1] // ratio + correction),
                            (self.head.x + ahead + self.probability_clock[Direction.RIGHT.value - 1],
                             self.head.y + correction),
                            (self.head.x + ahead,
                             self.head.y + self.probability_clock[Direction.RIGHT.value - 1] // ratio + correction)]

            color = WHITE if self.probability_clock[Direction.RIGHT.value - 1] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

        if self.direction == Direction.LEFT or Direction.UP or Direction.DOWN:
            arrow_points = [(self.head.x - ahead // 2,
                             self.head.y - self.probability_clock[Direction.LEFT.value] // ratio + correction),
                            (self.head.x - ahead // 2 - self.probability_clock[Direction.LEFT.value],
                             self.head.y + correction),
                            (self.head.x - ahead // 2,
                             self.head.y + self.probability_clock[Direction.LEFT.value] // ratio + correction)]

            color = WHITE if self.probability_clock[Direction.LEFT.value] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

        if self.direction == Direction.UP or Direction.LEFT or Direction.RIGHT:
            arrow_points = [(self.head.x + correction - self.probability_clock[Direction.UP.value] // ratio,
                             self.head.y - ahead // 2),
                            (self.head.x + correction,
                             self.head.y - self.probability_clock[Direction.UP.value] - ahead // 2),
                            (self.head.x + correction + self.probability_clock[Direction.UP.value] // ratio,
                             self.head.y - ahead // 2)]

            color = WHITE if self.probability_clock[Direction.UP.value] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

        if self.direction == Direction.DOWN or Direction.LEFT or Direction.RIGHT:
            arrow_points = [(self.head.x + correction - self.probability_clock[Direction.DOWN.value - 3] // ratio,
                             self.head.y + ahead),
                            (self.head.x + correction,
                             self.head.y + self.probability_clock[Direction.DOWN.value - 3] + ahead),
                            (self.head.x + correction + self.probability_clock[Direction.DOWN.value - 3] // ratio,
                             self.head.y + ahead)]

            color = WHITE if self.probability_clock[Direction.DOWN.value - 3] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.obstacle:
            pygame.draw.rect(self.display, BROWN,pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        snake_color_decay = AGENT_UI[self.agentID][0]
        decay_ratio = 0.90 if len(self.snake) < 10 else 0.97
        for pt in self.snake:
            pygame.draw.rect(self.display, snake_color_decay, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            snake_color_decay = (int(decay_ratio*snake_color_decay[0]),int(decay_ratio*snake_color_decay[1]),int(decay_ratio*snake_color_decay[2]))


        pygame.draw.rect(self.display, AGENT_UI[self.agentID][1],
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))


        if self.agentID:
            text = font.render(AGENT_NAMES[self.agentID-1]+" Score:" + str(self.score), True, AGENT_UI[self.agentID][0])
            self.display.blit(text, text_position[self.agentID-1])

        # user_control = font.render("Agent", True, WHITE)
        # self.display.blit(user_control, [self.w-80, 0])

        if self.arrow:
            self.arrow_ui()

        if not self.agentID:
            pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        if (type(action) != list) and (type(action) != np.ndarray):
            idx = action
            action = [0, 0, 0]
            action[idx] = 1
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        if self.arrow:
            self.probability_clock = [0,0,0,0]
            self.probability_clock[idx] = self.actions_probability[0]
            self.probability_clock[(idx + 1) % 4] = self.actions_probability[1]
            self.probability_clock[(idx - 1) % 4] = self.actions_probability[2]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
