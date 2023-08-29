# Visualizing RL: 
##### How the agent perceives the environment during training


#### Authors:
* [Yaad Rebhun](https://github.com/YaadR)
* [Itai Zeitouni](https://github.com/Itaize33)

## Project description:
Visualizing how different agents perceive their environment in the game Snake: Algorithms in Reinforcement Learning

### SNAKE:
The snake game environment is a visualization tool to evaluate different RL algorithms on the game Snake.
the algorithms need to learn how to guide the snake to the food without hitting the wall or eating itself (self-loop).
these algorithms do that by using image processing to create an input vector of values used in determining the best next step for the snake to play.
each time the snake eats the food the algorithm gets a reward.
the input vector has 3 values. each value defines the next step (forward, left, right). this is defined differently in each algorithm using state/action values.

### Project setups:
1. Run ```pip install -r requirements.txt```.
2. Every agent can run independently. The arena file is the only file that can also run independently. it runs all agents simultaneously in the same arena to compete and determine the best agent for the snake game. 

### Visualization solutions:
1. Heatmap
2. Certainty Arrows
3. Neural network weights visualization (where NN is used)
4. Certainty Bar (Entropy based)
5. State Activation Layer
 
<p float="center">
  <a href="https://youtu.be/ayUZIII4FsM">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/heatmap-giff.gif" width="350" />
  </a>
  <a href="https://youtu.be/oC77yQJnOHg">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/arrows-gif.gif" width="350" />
  </a>
</p>
<p float="center">
  <a href="https://youtu.be/tZ9F8GUXxgw">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/netviz-gif.gif" width="350" />
  </a>
  <a href="https://youtu.be/sAIO_uet1lw">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/certainty-giff.gif" width="350" />
  </a>
</p>
<p float="center">
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_17.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_34.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_129.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_322.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_534.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_584.jpg" width="125" />
</p>
<p float="center">
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_788.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_833.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_1673.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_1833.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_1930.jpg" width="125" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/valid%20activation/state_1810.jpg" width="125" />
</p>


## Reinforcement algorithms & concepts:
#### Agent State Value:
 - Value-based: state value
 - Model-based
 - off policy
 - online

RL Algorithm:
$$V(s_t)' = V(s_t) + \alpha* \left[ R_{t+1} + (1-s_{t->terminal})(\gamma* V(s_{t+1}) - V(s_t) \right)]$$


#### Agent Action Value:
 - Value-based: action value
 - Model-free
 - off policy
 - online

RL Algorithm:
$$Q(s_t)' = R_{t+1} + (1-s_{t->terminal})(\gamma* Max(Q(S_{t+1},A_{t+1})))$$


#### Agent Policy:
 - Policy-based
 - Model-free
 - on policy
 - online

RL Algorithm:
Critic:
$$A_{\text{critic}}(s_{t})' = R_{t} + (1-s_{t->terminal}) ( \gamma* V(s_{t+1}) - V(s_{t}) )$$

Actor:
$$\theta_{\text{actor}} \leftarrow \theta_{\text{actor}} +\nabla_{\theta_{\text{actor}}} \log(\pi_{\theta_{\text{actor}}}(a_{t}|s_{t})) A_{\text{critic}}(s_{t})$$





#### Training - Stability, Mean & STD - 20 Rounds:
<p float="left">
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/Buffers/Mean%20State%20Value%20Agents.jpg" width="250" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/Buffers/Mean%20Action%20Value%20Agents.jpg?raw=true" width="250" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/Buffers/Mean%20Action%20Policy%20Agents.jpg" width="250" />
</p>
<p float="left">
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/Buffers/STD%20State%20Value%20Agents.jpg" width="250" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/Buffers/STD%20Action%20Value%20Agents.jpg" width="250" />
  <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/plots/Buffers/STD%20Action%20Policy%20Agents.jpg" width="250" />
</p>

### Algorithms compete in an Arena
[![Arena](https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/arena-compete-gif.gif)](https://youtu.be/_RmOt_EeuUU)




## Additional Notes

Please follow the project steps carefully and ensure that all dependencies are correctly installed before running the solution.

## Acknowledgments
The basis for this project is inspired by [Patrick Loeber](https://github.com/patrickloeber) in his work at [Teach AI To Play Snake - Reinforcement Learning Tutorial With PyTorch And Pygame](https://youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)
## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it according to the terms of the license.

