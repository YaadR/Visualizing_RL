from game_multi import SnakeGameArena, SnakeGameAICompetition,pygame
import numpy as np
import matplotlib.pyplot as plt
import agent_Action_Value
import agent_Q
import agent_Q_Lambda
import agent_Value


BLACK = (0, 0, 0)
WIDTH = 480
HEIGHT = 360

TRAIN_STOP = 6
plot_colors = ['blue','green','purple']

def train(agent_num=1):
    trained = [False]*agent_num
    plot_scores = []
    for _ in range(agent_num):
        plot_scores.append([])
    plot_mean_scores = []
    for _ in range(agent_num):
        plot_mean_scores.append([])
    total_score = [0]*agent_num
    record = [0]*agent_num
    arena = SnakeGameArena(arrow=True,agent_num=agent_num)

    plt.ion()
    figure, axis = plt.subplots(agent_num, 1, figsize=(6, 5.5))
    plt.subplots_adjust(hspace=1)


    while True:
        # get old state
        for i,agent in enumerate(Agents):
            if not trained[i]:
                state = agent.get_state(arena.env[i])

                if i>1 and agent.array_tobinary(state) not in agent.Q.keys():
                    agent.Q[agent.array_tobinary(state)] = [0,0,0]
                    if i:
                        agent.eligibility_trace[agent.array_tobinary(state)] = [0, 0, 0]

                # get move
                action = agent.get_action(state)
                # print(agent.actions_probability, action)
                arena.env[i].actions_probability = agent.actions_probability

                # perform move and get new state
                reward, done, score = arena.env[i].play_step(action)
                arena.displays[i] = pygame.surfarray.array3d(arena.env[i].display)
                state_next = agent.get_state(arena.env[i])
                if i>1 and agent.array_tobinary(state_next) not in agent.Q.keys():
                    agent.Q[agent.array_tobinary(state_next)] = [0,0,0]
                    if i:
                        agent.eligibility_trace[agent.array_tobinary(state_next)] = [0, 0, 0]

                if i==2:
                    agent.update_Q(state,action,reward,state_next)
                elif i==1:
                    # perform move and get new state
                    _reward, _done, _state_next = agent.get_states_value(arena.env[i])

                    # train short memory
                    agent.train_short_memory(state, _reward, _state_next, _done)
                elif i==0:
                    # train short memory
                    agent.train_short_memory(state, action, reward, state_next, done)

                    # remember
                    agent.remember(state, action, reward, state_next, done)

                if done:
                    arena.env[i].reset()
                    agent.n_games += 1
                    if not i:
                        agent.train_long_memory()

                    if score > record[i]:
                        record[i] = score

                    plot_scores[i].append(score)
                    total_score[i] += score
                    mean_score = total_score[i] / agent.n_games
                    plot_mean_scores[i].append(mean_score)
                    trained[i] = mean_score > TRAIN_STOP



                    axis[i].cla()
                    axis[i].set_title(AGENT_NAMES[i])
                    axis[i].set_xlabel('Games')
                    axis[i].set_ylabel('Score')
                    axis[i].plot(plot_scores[i],color=plot_colors[i])
                    axis[i].plot(plot_mean_scores[i])
                    axis[i].axhline(y=TRAIN_STOP, color='orange', linestyle='--')
                    axis[i].set_ylim(ymin=0)
                    axis[i].text(len(plot_scores[i]) - 1, plot_scores[i][-1], str(plot_scores[i][-1]))
                    axis[i].text(len(plot_mean_scores[i]) - 1, plot_mean_scores[i][-1], str(plot_mean_scores[i][-1]))
                    plt.show(block=False)
                    plt.pause(.1)

                    arena.env[i].display.fill(BLACK)
                    if trained[i]:
                        arena.displays[i] = pygame.surfarray.array3d(arena.env[i].display)
                    print('Agent:',i,'Game:',agent.n_games, 'Score:', score, 'Record:', record[i], 'Mean Score:', round(mean_score,3))


        if np.all(trained):
            break
        arena.display.fill(BLACK)
        background = np.zeros((WIDTH,HEIGHT,3))
        for i in range(agent_num):
            background = np.where(background==0,arena.displays[i],background)
        surface = pygame.surfarray.make_surface(background)
        arena.display.blit(surface, (0, 0))
        pygame.display.flip()

def play(agent_num=1):
    plot_scores = []
    for _ in range(agent_num):
        plot_scores.append([])
    plot_mean_scores = []
    for _ in range(agent_num):
        plot_mean_scores.append([])
    total_score = [0]*agent_num
    record = [0]*agent_num
    arena = SnakeGameAICompetition(arrow=True,agent_num=agent_num,obstacle_flag=True)

    # plt.ion()
    # figure, axis = plt.subplots(3, 1, figsize=(6, 5.5))
    # plt.subplots_adjust(hspace=1)


    while True:
        # get old state
        for i,agent in enumerate(Agents):

            state = agent.get_state_arena(arena,i)
            # print(state)

            if i>1 and agent.array_tobinary(state) not in agent.Q.keys():
                agent.Q[agent.array_tobinary(state)] = np.random.dirichlet(np.ones(3))
                if i:
                    agent.eligibility_trace[agent.array_tobinary(state)] = np.random.dirichlet(np.ones(3))
            # get action
            action = agent.get_action(state)
            # arena.actions_probability[i] = agent.actions_probability

            # perform move and get new state
            _, done, score = arena.play_step(action,i)

            if done:
                arena.little_reset(i)
                agent.n_games += 1

                if score[i] > record[i]:
                    record[i] = score[i]

                plot_scores[i].append(score[i])
                total_score[i] += score[i]
                mean_score = total_score[i] / agent.n_games
                plot_mean_scores[i].append(mean_score)

                # arena.display.fill(BLACK)
                print('Agent:',i,'Game:',agent.n_games, 'Score:', score[i], 'Record:', record[i], 'Mean Score:', round(mean_score,3))



        pygame.display.flip()
        arena.display.fill(BLACK)


if __name__ == '__main__':
    # AGENT_NAMES = ["Action Value","Value Based","Q(Lambda)"]
    AGENT_NAMES = ["Action Value","Value Based"]
    # AGENT_NAMES = ["Action Value"]

    agentActionValue = agent_Action_Value.Action_Value()
    agent_Q = agent_Value.Agent_Value()
    # agent_Q_Lambda = agent_Q_Lambda.Agent_Q_Lambda()
    # Agents = [agent_Action_Value,agent_Q,agent_Q_Lambda]
    Agents = [agent_Action_Value,agent_Q]
    # Agents = [agent_Action_Value]
    train(agent_num=len(Agents))
    # train()
    plt.close()
    # pygame.quit()
    # pygame.init()
    play(agent_num=len(Agents))
    # play()