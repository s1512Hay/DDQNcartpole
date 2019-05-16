#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:26:51 2019

@author: shay
"""
"""
Environment solved- CartPole-v1
Method: DQN (38-40 Lecture 6- Value Function Approximation, David Silver)
Code flow- Class CartPoleAgent holds all the necessary information in order to preserve agents experience. 
Upon execution an instance of the class is produced if a previous agent stored as a pickle file is not in the current directory. The agent is trained with double deep q-networks with experience replay- one policy network that is being constantly updated (once there are enough transitions in memory replay) and one target network that is being updated occasionally (ti.e every untill_update episodes)
On initialization using build_net function we construct 2 identical neural networks with the following architecture- 
first layer: 4(state dimension)→24 classic linear with rectified linear unit activation
second layer: 24→ 24 classic linear with linear unit activation
output: full-connected linear  24→2 output layer
We use Adam optimizer and MSE as the loss to be minimized.
"""
import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle



class CartPoleAgent:
    
    def __init__(self, env):
        self.env = env
        self.num_epsiodes = 3000
        self.replay_memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0  
        self.min_epsilon = 0.01
        self.decay = 0.998
        self.alpha = 0.0005
        self.policy_net = self.build_net()
        self.target_net = self.build_net()
        self.till_update = 10
        self.batch_size =32
        
        """This is the constructor for both policy and target nets.
            both use identical classsic 4-> 24 (with bias variable) linear layer,
            with recitified linear unit as activation function, another 24->24 clasic linear layer
            and a fully connected 24-> 2 output layer. Adam ptimizer is an extension for SGD (to read more:
            https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
        """
    def build_net(self):
        net = Sequential()
        net.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu'))
        net.add(Dense(24, activation='relu'))
        net.add(Dense(2, activation='linear'))
        net.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return net
    """
    procedure that copies the weights from the policy net to the target net.
    since both have the exact smae structure this is equivalant to replicating the policy net, which is done whenever
    we want to update the target net.
    """
    def update_target_net(self):
        W = self.policy_net.get_weights()
        self.target_net.set_weights(W)
    """
    Epsilon greedy algorithm with vanishing epsilon ( in order to exploit more in later stages of training/
    explore more in the beginning)
    """     
    def qDerivedPolicy(self, state):
        if (random.uniform(0, 1) < self.epsilon):
            return self.env.action_space.sample()
        else:
            qsas = self.policy_net.predict(state)
            return np.argmax(qsas[0])
    """
    Optimization step- For a batch of transitions sampled from the replay memory of the agent,
    and for all transitions (s,a,r,s') the approximation of Q(S,A) is computed from the policy net and the target value 
    is learnt from target net
    """
    def optimize(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                target = reward + self.gamma *np.amax(self.target_net.predict(next_state)[0])
            else:
                target = reward
            target_vec = self.policy_net.predict(state) #this is a trick so that only relevant weights will be updated (ones corresponding to the action taken)
            target_vec[0][action] = target
            self.policy_net.fit(state, target_vec, epochs=1, verbose=0)#preform one single epoch towards target
    #this simply converts ndarry state to [state] shape array to fit the network input dimensions (as a batch of 1)
    def input_form(self, state):
        state_first_dim = self.env.observation_space.shape[0]
        return np.reshape(state, [1, state_first_dim])
            
    """
    main learning loop- transiotions are saved in replay memory, once there are enough transitions we optimize 
    the error with respect to fixed Q values obtained from the target net, and update the target net every till_update
    episodes.
    """    
    def train(self):
        durations = []
        avg_durations = []
        for episode in range(self.num_epsiodes):
            state = self.env.reset()
            state = self.input_form(state)
            duration = 0
            done = False
            while not done:
                # env.render() #uncomment to display
                duration+=1
                action = self.qDerivedPolicy(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.input_form(next_state)
                if done:
                    durations.append(duration)
                    reward = -10
                self.replay_memory.append((state, action, reward, next_state, done))
                state = next_state
            if len(self.replay_memory) > self.batch_size:
                self.optimize()
                self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)#we update epsilon after every batch update
            if episode % self.till_update == 0:
                self.update_target_net()#update the target net - copy policy nets weights
            if episode % 500 == 0 and episode is not 0:
                avg_durations.append(sum(durations)/episode )                    
        plt.plot(avg_durations)
        plt.title('Average duration rate over time')
        plt.xlabel('Time units of 500 episodes')
        plt.ylabel('Average duration rate')
        plt.show('Average duration rates')
        
        """
   100 consecutive episodes of policy derived action selection.
   """
    def test(self):
        overall_duration = 0
        for episode in range(100):
            state = self.env.reset()
            state = self.input_form(state)
            done = False
            while not done:
                #self.env.render() #uncommnet to display
                overall_duration+=1
                action = np.argmax(self.policy_net.predict(state)[0])
                next_state, reward, done, info = self.env.step(action)
                next_state = self.input_form(next_state)
                state = next_state    
        print("Average duration over 100 consecutive episodes: " + str(overall_duration/100))
"""
this is where program begins to run if being ran from this page.
An existing agent in form of pickle file is searched for in current directory, if one is not found then a new instance of 
an agent is created, trained and tested.
"""           
def main():
    try:
        agent = pickle.load( open( "myCartPoleAgent.p", "rb" ) )
        agent.test()
    except (OSError, IOError) :
        env = gym.make("CartPole-v1")
        env.seed(1992)
        agent= CartPoleAgent(env)
        agent.train()
        agent.test()
        pickle.dump( agent, open( "myCartPoleAgent.p", "wb" ) )
    
if __name__ == "__main__":
    main()