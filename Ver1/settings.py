class settings:
    BLOCK_SIZE = 20
    WIDTH = 480
    HEIGHT = 360

    MAX_MEMORY = 100_000
    HIDDEN_LAYER = 256
    LR = 0.001
    STATE_VEC_SIZE = 11
    NUM_ACTIONS = 3
    USE_HEAT_MAP = False
    ALPHA = 0.1
    GAMMA = 0.9
    STATE_VALUE = 1
    EPSILON = 50
    NUM_EPISODES = 100
    BATCH_SIZE = 1000
    LAMBDA = 0.8
    SPEED = 20

    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)


S = settings()
