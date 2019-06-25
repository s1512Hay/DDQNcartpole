# DDQNcartpole
DDQN solver for cartpole (modification)
Environment solved- CartPole-v1
Method: DQN (38-40 Lecture 6- Value Function Approximation, David Silver)
Code flow- Class CartPoleAgent holds all the necessary information in order to preserve agents experience. 
Upon execution an instance of the class is produced if a previous agent stored as a pickle file is not in the current directory. The agent is trained with double deep q-networks with experience replay- one policy network that is being constantly updated (once there are enough transitions in memory replay) and one target network that is being updated occasionally (ti.e every untill_update episodes)
On initialization using build_net function we construct 2 identical neural networks with the following architecture- 
first layer: 4(state dimension)→24 classic linear with rectified linear unit activation
second layer: 24→ 24 classic linear with linear unit activation
output: full-connected linear  24→2 output layer
We use Adam optimizer and MSE as the loss to be minimized.
