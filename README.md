# Visualizing RL: 
##### How the agent perceive the environment during training


#### Authors:
* [Yaad Rebhun](https://github.com/YaadR)
* [Itai Zeitouni](https://github.com/Itaize33)

## Project description:
Visualizing how different agents perceives their environment in the game Snake : algorithms in Reinforcement Learning

### Project set ups:

### Visualization solutions:
1. Heat Map
2. Certainty Arrows
3. Neural network weights visualization (where NN is used)
4. 
## Reinforcement algorithms & concepts:
#### Agent State Value:
 - Value based : state value
 - Model based
 - off policy
 - online
   
The Bellman equation:
$$V(s) = \max_a \left( R(s,a) + \gamma \sum_{s'} P(s' | s,a) V(s') \right)$$
Iterative form:
$$V(S_t) = V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

#### Agent Action Value:
 - Value based : action value
 - Model free
 - off policy
 - online


#### Agent Policy:
 - Policy based
 - Model free
 - on policy
 - online



### Algorithms compete in an Arena



## Additional Notes

Please follow the project steps carefully and ensure that all dependencies are correctly installed before running the solution.

## Disclaimer: 
This README provides an overview of the project and its approach to_____. For in-depth instructions and code implementations, refer to the source code and accompanying documentation in the repository.

## Acknowledgments
The basis for this project is inspired by [Patrick Loeber](https://github.com/patrickloeber) in his work at [Teach AI To Play Snake - Reinforcement Learning Tutorial With PyTorch And Pygame](https://youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)
## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it according to the terms of the license.

