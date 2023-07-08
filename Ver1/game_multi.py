import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from pygame.locals import *
from game import SnakeGameAI

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
GREEN = (0, 255, 0)
GREEN_FOOD = (0, 150, 0)
PURPLE = (128, 0, 128)
PURPLE_FOOD = (224, 0, 224)
YELLOW = (128, 0, 128)
BROWN = (139, 69, 19)


BLOCK_SIZE = 20
WIDTH = 480
HEIGHT = 360
SPEED = 30

AGENT_UI = [[BLUE, RED], [BLUE, BLUE_FOOD], [GREEN, GREEN_FOOD], [PURPLE, PURPLE_FOOD]]

OBSTACLE_HEIGHT = 60
OBSTACLE_WIDTH = 80

class SnakeGameArena:
    def __init__(self, w=WIDTH, h=HEIGHT, arrow=False, agent_num=1):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((0,0),pygame.FULLSCREEN) # (self.w, self.h)

        self.displays = [None]*agent_num
        self.env = []
        for i in range(1,agent_num+1):
            self.env.append(SnakeGameAI(arrow=arrow, agentID=i))

text_position = [[0, 0],[0, 25],[0, 50]]
AGENT_NAMES = ["DQN","TD(0)","TD(Lambda)"]

class SnakeGameAICompetition:

    def __init__(self, w=WIDTH, h=HEIGHT, arrow=False, agentID=0,agent_num=1,obstacle_flag=False):
        self.w = w
        self.h = h
        self.arrow = arrow
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Competition')
        self.clock = pygame.time.Clock()
        self.agent_num = agent_num
        self.obst_flag = obstacle_flag
        self.reset()
        self.actions_probability = [[0, 0, 0]]*agent_num
        self.probability_clock = [[0, 0, 0, 0]]*agent_num
        self.agentID = agentID

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
            self.place_obstacle()  ###
        self.direction = []
        self.head = []
        self.snake = []
        for i in range(self.agent_num):
            self.direction.append(Direction.RIGHT)

            self.head.append(Point(self.w / 2, self.h / 2))
            self.snake.append([self.head[i],
                          Point(self.head[i].x - BLOCK_SIZE, self.head[i].y),
                          Point(self.head[i].x - (2 * BLOCK_SIZE), self.head[i].y)])

        self.score = [0]*self.agent_num
        self.food = None #[None]*self.agent_num
        self._place_food()
        self.frame_iteration = [0]*self.agent_num

    def little_reset(self,agentID=0):

        self.direction[agentID] = (Direction.RIGHT)

        self.head[agentID] = (Point(self.w / 2, self.h / 2))
        self.snake[agentID] =[self.head[agentID],
                      Point(self.head[agentID].x - BLOCK_SIZE, self.head[agentID].y),
                      Point(self.head[agentID].x - (2 * BLOCK_SIZE), self.head[agentID].y)]

        # self.score[agentID] = 0
        self.frame_iteration[agentID] = 0



    def _place_food(self,agentID=0):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake[agentID] or self.food in self.obstacle:
            self._place_food()

    def play_step(self, action,agentID=0):
        self.frame_iteration[agentID] += 1

        self._move(action,agentID)  # update the head
        self.snake[agentID].insert(0, self.head[agentID])

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision(agentID=agentID) or self.frame_iteration[agentID] > 50 * len(self.snake[agentID]):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head[agentID] == self.food:
            self.score[agentID] += 1
            reward = 10
            self._place_food(agentID=agentID)
        else:
            self.snake[agentID].pop()

        # 5. update ui and clock
        self._update_ui(agentID)

        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None,agentID=0):
        if pt is None:
            pt = self.head[agentID]
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[agentID][1:]:
            return True

        if pt in self.obstacle:
            return True

        return False


    def arrow_ui(self,agentID=0):

        if np.any(self.probability_clock[agentID]):
            self.probability_clock[agentID] = (self.probability_clock[agentID] - np.min(self.probability_clock[agentID])) / (
                        np.max(self.probability_clock[agentID]) - np.min(self.probability_clock[agentID]))
        self.probability_clock[agentID] *= 20
        correction = BLOCK_SIZE // 2
        ahead = 30
        ratio = 3

        if self.direction[agentID] == Direction.RIGHT or Direction.UP or Direction.DOWN:
            arrow_points = [(self.head[agentID].x + ahead,
                             self.head[agentID].y - self.probability_clock[agentID][Direction.RIGHT.value - 1] // ratio + correction),
                            (self.head[agentID].x + ahead + self.probability_clock[agentID][Direction.RIGHT.value - 1],
                             self.head[agentID].y + correction),
                            (self.head[agentID].x + ahead,
                             self.head[agentID].y + self.probability_clock[agentID][Direction.RIGHT.value - 1] // ratio + correction)]

            color = WHITE if self.probability_clock[agentID][Direction.RIGHT.value - 1] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

        if self.direction[agentID] == Direction.LEFT or Direction.UP or Direction.DOWN:
            arrow_points = [(self.head[agentID].x - ahead // 2,
                             self.head[agentID].y - self.probability_clock[agentID][Direction.LEFT.value] // ratio + correction),
                            (self.head[agentID].x - ahead // 2 - self.probability_clock[agentID][Direction.LEFT.value],
                             self.head[agentID].y + correction),
                            (self.head[agentID].x - ahead // 2,
                             self.head[agentID].y + self.probability_clock[agentID][Direction.LEFT.value] // ratio + correction)]

            color = WHITE if self.probability_clock[agentID][Direction.LEFT.value] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

        if self.direction[agentID] == Direction.UP or Direction.LEFT or Direction.RIGHT:
            arrow_points = [(self.head[agentID].x + correction - self.probability_clock[agentID][Direction.UP.value] // ratio,
                             self.head[agentID].y - ahead // 2),
                            (self.head[agentID].x + correction,
                             self.head[agentID].y - self.probability_clock[agentID][Direction.UP.value] - ahead // 2),
                            (self.head[agentID].x + correction + self.probability_clock[agentID][Direction.UP.value] // ratio,
                             self.head[agentID].y - ahead // 2)]

            color = WHITE if self.probability_clock[agentID][Direction.UP.value] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

        if self.direction[agentID] == Direction.DOWN or Direction.LEFT or Direction.RIGHT:
            arrow_points = [(self.head[agentID].x + correction - self.probability_clock[agentID][Direction.DOWN.value - 3] // ratio,
                             self.head[agentID].y + ahead),
                            (self.head[agentID].x + correction,
                             self.head[agentID].y + self.probability_clock[agentID][Direction.DOWN.value - 3] + ahead),
                            (self.head[agentID].x + correction + self.probability_clock[agentID][Direction.DOWN.value - 3] // ratio,
                             self.head[agentID].y + ahead)]

            color = WHITE if self.probability_clock[agentID][Direction.DOWN.value - 3] else BLACK
            pygame.draw.polygon(self.display, color, arrow_points)

    def _update_ui(self, agentID=0):
        # self.display.fill(BLACK)

        for pt in self.obstacle:
            pygame.draw.rect(self.display, BROWN,pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake[agentID]:
            pygame.draw.rect(self.display, AGENT_UI[agentID+1][0], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))


        pygame.draw.rect(self.display, AGENT_UI[0][1],
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))


        text = font.render(AGENT_NAMES[agentID]+" Score:" + str(self.score[agentID]), True, AGENT_UI[agentID+1][0])
        self.display.blit(text, text_position[agentID])


        # user_control = font.render("Agent", True, WHITE)
        # self.display.blit(user_control, [self.w-80, 0])
        if self.arrow:
            self.arrow_ui(agentID=agentID)


        # pygame.display.flip()

    def _move(self, action,agentID=0):
        # [straight, right, left]
        if (type(action) != list) and (type(action) != np.ndarray):
            idx = action
            action = [0, 0, 0]
            action[idx] = 1
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction[agentID])

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        if self.arrow:
            self.probability_clock[agentID] = [0] * 4
            self.probability_clock[agentID][idx] = self.actions_probability[0]
            self.probability_clock[agentID][(idx + 1) % 4] = self.actions_probability[1]
            self.probability_clock[agentID][(idx - 1) % 4] = self.actions_probability[2]

        self.direction[agentID] = new_dir

        x = self.head[agentID].x
        y = self.head[agentID].y
        if self.direction[agentID] == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction[agentID] == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction[agentID] == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction[agentID] == Direction.UP:
            y -= BLOCK_SIZE

        self.head[agentID] = Point(x, y)
