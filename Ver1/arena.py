from game_multi import SnakeGameArena, SnakeGameAICompetition, pygame
import numpy as np
import matplotlib.pyplot as plt
import agent_Action_Value
import agent_Policy
import agent_Value
from settings import S

S.TRAIN_STOP = 15
S.MAX_GAMES = 300
S.MIN_GAMES = 150
S.PLOT_COLORS = ["blue", "purple", "green"]


def train(Agents, agent_num=1):
    trained = [False] * agent_num
    plot_scores = []
    for _ in range(agent_num):
        plot_scores.append([])
    plot_mean_scores = []
    for _ in range(agent_num):
        plot_mean_scores.append([])
    total_score = [0] * agent_num
    record = [0] * agent_num
    arena = SnakeGameArena(arrow=True, agent_num=agent_num)

    plt.ion()
    figure, axis = plt.subplots(agent_num, 1, figsize=(6, 5.5))
    plt.subplots_adjust(hspace=1)

    while True:
        # get old state
        for i, agent in enumerate(Agents):
            if not trained[i]:
                state = agent.get_state(arena.env[i])

                # get move
                action = agent.get_action(state)
                arena.env[i].actions_probability = agent.prediction

                if i == 2:
                    # perform moves in model and evaluate states value
                    _reward, _done, _state_next = agent.get_states_value(arena.env[i])
                    # train online model based
                    agent.train_online(state, _reward, _state_next, _done)

                # perform move and get new state
                reward, done, score = arena.env[i].play_step(action)
                arena.displays[i] = pygame.surfarray.array3d(arena.env[i].display)
                state_next = agent.get_state(arena.env[i])

                if i == 1:
                    # train online
                    agent.train_online(
                        state, np.argmax(action), reward, state_next, done
                    )
                elif i == 0:
                    agent.train_online(state, action, reward, state_next, done)
                if done:
                    arena.env[i].reset()
                    agent.n_games += 1

                    if score > record[i]:
                        record[i] = score

                    plot_scores[i].append(score)
                    total_score[i] += score
                    mean_score = total_score[i] / agent.n_games
                    plot_mean_scores[i].append(round(mean_score, 3))
                    trained[i] = (
                        (mean_score > S.TRAIN_STOP) and (agent.n_games >= S.MIN_GAMES)
                    ) or (agent.n_games >= S.MAX_GAMES)

                    axis[i].cla()
                    axis[i].set_title(S.AGENT_NAMES[i])
                    axis[i].set_xlabel("Games")
                    axis[i].set_ylabel("Score")
                    axis[i].plot(plot_scores[i], color=S.PLOT_COLORS[i])
                    axis[i].plot(plot_mean_scores[i])
                    axis[i].axhline(y=S.TRAIN_STOP, color="orange", linestyle="--")
                    if agent.n_games > S.MIN_GAMES - 10:
                        axis[i].axvline(x=S.MIN_GAMES, color="green", linestyle="--")
                    if agent.n_games > S.MAX_GAMES - 20:
                        axis[i].axvline(x=S.MIN_GAMES, color="red", linestyle="--")
                    axis[i].set_ylim(ymin=0)
                    axis[i].text(
                        len(plot_scores[i]) - 1,
                        plot_scores[i][-1],
                        str(plot_scores[i][-1]),
                    )
                    axis[i].text(
                        len(plot_mean_scores[i]) - 1,
                        plot_mean_scores[i][-1],
                        str(plot_mean_scores[i][-1]),
                    )
                    plt.show(block=False)
                    plt.pause(0.1)

                    arena.env[i].display.fill(S.BLACK)
                    if trained[i]:
                        arena.displays[i] = pygame.surfarray.array3d(
                            arena.env[i].display
                        )
                    print(
                        "Agent:",
                        i,
                        "Game:",
                        agent.n_games,
                        "Score:",
                        score,
                        "Record:",
                        record[i],
                        "Mean Score:",
                        round(mean_score, 3),
                    )

        if np.all(trained):
            break
        arena.display.fill(S.BLACK)
        background = np.zeros((S.WIDTH, S.HEIGHT, 3))
        for i in range(agent_num):
            background = np.where(background == 0, arena.displays[i], background)
        surface = pygame.surfarray.make_surface(background)
        arena.display.blit(surface, (0, 0))
        pygame.display.flip()


def play(Agents, agent_num=1):
    plot_scores = []
    for _ in range(agent_num):
        plot_scores.append([])
    plot_mean_scores = []
    for _ in range(agent_num):
        plot_mean_scores.append([])
    total_score = [0] * agent_num
    record = [0] * agent_num
    arena = SnakeGameAICompetition(arrow=False, agent_num=agent_num, obstacle_flag=True)

    while True:
        # get old state
        for i, agent in enumerate(Agents):
            state = agent.get_state_arena(arena, i)

            # get action
            action = agent.get_action(state)

            # perform move and get new state
            _, done, score = arena.play_step(action, i)

            if done:
                arena.little_reset(i)
                agent.n_games += 1

                if score[i] > record[i]:
                    record[i] = score[i]

                plot_scores[i].append(score[i])
                total_score[i] += score[i]
                mean_score = total_score[i] / agent.n_games
                plot_mean_scores[i].append(round(mean_score, 3))

                print(
                    "Agent:",
                    i,
                    "Game:",
                    agent.n_games,
                    "Score:",
                    score[i],
                    "Record:",
                    record[i],
                    "Mean Score:",
                    round(mean_score, 3),
                )

        pygame.display.flip()
        arena.display.fill(S.BLACK)


def main():
    S.AGENT_NAMES = ["Action Value", "Policy", "State Value"]
    agentActionValue = agent_Action_Value.Action_Value()
    agentA2C = agent_Policy.Agent_Policy()
    agentValue = agent_Value.Agent_Value()
    Agents = [agentActionValue, agentA2C, agentValue]
    train(Agents, agent_num=len(Agents))
    plt.close()
    play(Agents, agent_num=len(Agents))


if __name__ == "__main__":
    main()
