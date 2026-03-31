import sys
import os
import time
import numpy as np

# --- FOOLPROOF PATH FIX ---
# 1. Get the absolute path of the folder this script is in ('gym')
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the parent folder ('reinforcement-learning')
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent folder to Python's system path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 4. Import the class ABSOLUTELY (Notice there are NO dots before 'mdp')
from mdp.tabular import TabularMDP 
import toy_text as tt

env = input("Choose an environment (FrozenLake[1], CliffWalking[2], Taxi[3]): ")
match env:
    case "1":
        env = tt.create_environment("FrozenLake-v1", render_mode='human') 
    case "2":
        env = tt.create_environment("CliffWalking-v0", render_mode='human')
    case "3":
        env = tt.create_environment("Taxi-v3", render_mode='human')
    case _:
        print("Invalid choice. Defaulting to FrozenLake.")
        env = tt.create_environment("FrozenLake-v1", render_mode='human')

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

process = input("Choose a process (Value Iteration[1], Policy Iteration[2]): ")
match process:
    case "1":
        policy, V = mdp.value_iteration()
    case "2":
        policy, V = mdp.policy_iteration()
    case _:
        print("Invalid choice. Defaulting to Value Iteration.")
        policy, V = mdp.value_iteration()

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