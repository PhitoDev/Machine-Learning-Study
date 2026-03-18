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

# 4. Import the class ABSOLUTELY (Notice there are NO dots before 'mdp')
from mdp.mdp import TabularMDP 

env = gym.make("FrozenLake-v1", map_name="8x8", render_mode='human', is_slippery=False)
num_states = env.observation_space.n
num_actions = env.action_space.n
gamma = 0.99
P = np.zeros((num_states, num_actions, num_states))
R = np.zeros((num_states, num_actions))

for s in range(num_states):
    for a in range(num_actions):
        outcomes = env.unwrapped.P[s][a]

        for prob, next_state, reward, terminated in outcomes:
            P[s, a, next_state] += prob
            R[s, a] = prob * reward

mdp = TabularMDP(
    num_states=num_states, 
    num_actions=num_actions, 
    gamma=gamma, 
    P=P, 
    R=R
)

policy, V = mdp.policy_iteration()

state, info = env.reset()
terminated = False
truncated = False

env.render()
while not terminated and not truncated:
    
    action = policy[state] # Always take the best known action
    state, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.2)

os.system('clear')
#env.render()
if reward == 1:
    print("Goal reached! 🚩")
else:
    print("Agent fell in a hole. 🕳️")

env.close()