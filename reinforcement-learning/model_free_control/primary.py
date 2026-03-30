import numpy as np
from enum import Enum

class ControlAlgorithm(Enum):
   MonteCarlo = 1
   QLearning = 2
   SARSA = 3

class ModelFreeControl:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((self.num_states, self.num_actions))

    def update_monte_carlo(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            self.Q[state, action] += self.alpha * (G - self.Q[state, action])

    def update_sarsa(self, state, action, reward, next_state, next_action):
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                
    def update_q_learning(self, state, action, reward, next_state):
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])