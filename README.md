# RL_Maze_Solving

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/204293537-90708642-667a-4dac-8633-1dc40f8631db.png" height="300" title="Example Maze and learned Policy">
</p>
<p align="center">
  <em>Easy example Maze and learned Policy</em>
</p>

This repository creates a random maze and attempts to solve it using Deep Q-learning. The agent may not hold any memory other than the experience replay buffer, may not import anything from the environment, and may not use any heuristics to solve the maze - this is a pure Deep Q-learning solution.
### Random_environment.py
Creates a random maze to be solved, draws it and the policy, sets the rules
### Train_and_test.py
Trains the DQN for 10 minutes and then tests it
### Agent.py
Chooses actions, contains DQN and experience replay buffer
