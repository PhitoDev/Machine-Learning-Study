import sys
import os
import time
import numpy as np
import gymnasium as gym
# --- FOOLPROOF PATH FIX ---
# 1. Get the absolute path of the folder this script is in ('gym')
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the parent folder ('reinforcement-learning')
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent folder to Python's system path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import model_free_control.primary as mfc

env = gym.make("FrozenLake-v1", map_name="8x8", render_mode='human', is_slippery=True)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999
num_states = env.observation_space.n
num_actions = env.action_space.n
gamma = 0.99
alpha = 0.1
control = mfc.ModelFreeControl(num_states, num_actions, alpha, gamma)

print("Training")
algorithm = input("Choose an algorithm (MonteCarlo[1], SARSA[2], QLearning[3]): ")
match algorithm:
    case "1":
        algorithm = mfc.ControlAlgorithm.MonteCarlo
    case "2":
        algorithm = mfc.ControlAlgorithm.SARSA
    case "3":
        algorithm = mfc.ControlAlgorithm.QLearning
    case _:
        algorithm = mfc.ControlAlgorithm.MonteCarlo
        print("Invalid choice. Defaulting to Monte Carlo.")

wins = 0
num_episodes = 10000

for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_data = []
    
    if algorithm == mfc.ControlAlgorithm.SARSA:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(control.Q[state])

    while not terminated and not truncated:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
             action = np.argmax(control.Q[state])   
            
        new_state, reward, terminated, truncated, info = env.step(action)

        match algorithm:
            case mfc.ControlAlgorithm.SARSA:
                if np.random.uniform(0, 1) < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(control.Q[new_state])
                control.update_sarsa(state, action, reward, new_state, next_action)
                action = next_action
            case mfc.ControlAlgorithm.QLearning:
                control.update_q_learning(state, action, reward, new_state)
                episode_data.append((state, action, reward, new_state))
            case mfc.ControlAlgorithm.MonteCarlo:
                episode_data.append((state, action, reward))
        state = new_state

        if reward == 1:
            wins += 1

    if algorithm == mfc.ControlAlgorithm.MonteCarlo:    
        control.update_monte_carlo(episode_data)

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
    action = np.argmax(control.Q[state])
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)

os.system('clear')
if reward == 1:
    print("Goal reached! 🚩")
else:
    print("Agent fell in a hole. 🕳️")
env.close()