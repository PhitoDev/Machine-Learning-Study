import gymnasium as gym
import numpy as np
import os
import sys
import time

# 1. Setup Environment
# 'ansi' mode returns a string that we can print to the terminal
env = gym.make("FrozenLake-v1", render_mode='ansi', map_name="8x8", is_slippery=False) 

# Hyperparameters
total_episodes = 10000
Q = np.zeros((env.observation_space.n, env.action_space.n))
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999
alpha = 0.1
gamma = 0.99

wins = 0

# The Core Q-Learning Loop
print("Training")
for episode in range(total_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    
    while not terminated and not truncated:
        
        # 1. Choose Action (Epsilon-Greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q[state])       
            
        # 2. Take Action
        new_state, reward, terminated, truncated, info = env.step(action)
        
        # 3. Update Q-Table
        # Formula: Q[s,a] = Q[s,a] + alpha * (reward + gamma * max(Q[s',a']) - Q[s,a])
        best_next_action = np.max(Q[new_state])
        td_target = reward + gamma * best_next_action
        Q[state, action] += alpha * (td_target - Q[state, action])
        
        # 4. Move to Next State
        state = new_state
        
        if reward == 1:
            wins += 1
    # Decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()

print("Training finished!")
time.sleep(0.5)

if wins == 0:
    print("Agent could not find a path. Try again.")
    sys.exit()

# --- WATCHING THE AGENT ---
env = gym.make("FrozenLake-v1", map_name="8x8", render_mode='human', is_slippery=False)
state, info = env.reset()
terminated = False
truncated = False

env.render()
while not terminated and not truncated:
    
    action = np.argmax(Q[state]) # Always take the best known action
    state, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.2)

os.system('clear')
#env.render()
if reward == 1:
    print("Goal reached! 🚩")
else:
    print("Agent fell in a hole. 🕳️")

env.close()