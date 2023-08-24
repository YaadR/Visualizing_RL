# Visualizing RL: 
##### How the agent perceive the environment during training


#### Authors:
* [Yaad Rebhun](https://github.com/YaadR)
* [Itai Zeitouni](https://github.com/Itaize33)

## Project description:
Visualizing how different agents perceives their environment in the game Snake : algorithms in Reinforcement Learning

### Project set ups:

### Visualization solutions:
1. Heatmap
2. Certainty Arrows
3. Neural network weights visualization (where NN is used)
4. Certainty Bar (Entropy based)
 
<p float="left">
  <a href="https://youtu.be/ayUZIII4FsM">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/heatmap-giff.gif" width="350" />
  </a>
  <a href="https://youtu.be/link-to-second-video">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/arena-compete-gif.gif" width="350" />
  </a>
</p>
<p float="left">
  <a href="https://youtu.be/link-to-second-video">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/arena-compete-gif.gif" width="350" />
  </a>
  <a href="https://youtu.be/sAIO_uet1lw">
    <img src="https://github.com/YaadR/Visualizing_RL/blob/main/Ver1/data/materials/certainty-giff.gif" width="350" />
  </a>
</p>



## Reinforcement algorithms & concepts:
#### Agent State Value:
 - Value based : state value
 - Model based
 - off policy
 - online

RL Algorithm:
$$V(S_t)' = V(S_t) + \alpha* \left[ R_{t+1} + (1-S_{t->terminal})(\gamma* V(S_{t+1}) - V(S_t) \right)]$$


#### Agent Action Value:
 - Value based : action value
 - Model free
 - off policy
 - online

RL Algorithm:
$$Q(S_t)' = R_{t+1} + (1-S_{t->terminal})(\gamma* Max(Q(S_{t+1},A_{t+1})))$$


#### Agent Policy:
 - Policy based
 - Model free
 - on policy
 - online

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

## Disclaimer: 
This README provides an overview of the project and its approach to_____. For in-depth instructions and code implementations, refer to the source code and accompanying documentation in the repository.

## Acknowledgments
The basis for this project is inspired by [Patrick Loeber](https://github.com/patrickloeber) in his work at [Teach AI To Play Snake - Reinforcement Learning Tutorial With PyTorch And Pygame](https://youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)
## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it according to the terms of the license.

