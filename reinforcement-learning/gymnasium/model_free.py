import os
import sys
import time

import numpy as np

# --- FOOLPROOF PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import toy_text as tt
from phito_rl.tabular.modelfree import ControlAlgorithm, ModelFreeControl

env_input = input("Choose an environment (FrozenLake[1], CliffWalking[2], Taxi[3]): ")
match env_input:
    case "1":
        env = tt.create_environment("FrozenLake-v1", render_mode="ansi")
    case "2":
        env = tt.create_environment("CliffWalking-v1", render_mode="ansi")
    case "3":
        env = tt.create_environment("Taxi-v3", render_mode="ansi")
    case _:
        print("Invalid choice. Defaulting to FrozenLake.")
        env = tt.create_environment("FrozenLake-v1", render_mode="ansi")

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
num_states = env.observation_space.n
num_actions = env.action_space.n
gamma = 0.99
alpha = 0.001
control = ModelFreeControl(num_states, num_actions, alpha, gamma)

print("Training")
algorithm = input("Choose an algorithm (MonteCarlo[1], SARSA[2], QLearning[3]): ")
match algorithm:
    case "1":
        algorithm = ControlAlgorithm.MonteCarlo
    case "2":
        algorithm = ControlAlgorithm.SARSA
    case "3":
        algorithm = ControlAlgorithm.QLearning
    case _:
        algorithm = ControlAlgorithm.MonteCarlo
        print("Invalid choice. Defaulting to Monte Carlo.")

rng = np.random.default_rng(42)
wins = 0
num_episodes = int(
    input("Enter the number of episodes for training (e.g., 10000): ") or "10000"
)

for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_data = []

    if algorithm == ControlAlgorithm.SARSA:
        if rng.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(control.Q[state])

    while not terminated and not truncated:
        if rng.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(control.Q[state])

        new_state, reward, terminated, truncated, info = env.step(action)

        match algorithm:
            case ControlAlgorithm.SARSA:
                if rng.uniform(0, 1) < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(control.Q[new_state])
                control.update_sarsa(state, action, reward, new_state, next_action)
                action = next_action
            case ControlAlgorithm.QLearning:
                control.update_q_learning(state, action, reward, new_state)
            case ControlAlgorithm.MonteCarlo:
                episode_data.append((state, action, reward))
        state = new_state

        if reward == 1:
            wins += 1

    if algorithm == ControlAlgorithm.MonteCarlo:
        control.update_monte_carlo(episode_data)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()

print("Training finished!")
time.sleep(0.5)

if wins == 0:
    os.system("clear")
    print("Agent could not find a path. Try again.")
    sys.exit()


# --- WATCHING THE AGENT ---
match env_input:
    case "1":
        env = tt.create_environment("FrozenLake-v1", render_mode="human")
    case "2":
        env = tt.create_environment("CliffWalking-v1", render_mode="human")
    case "3":
        env = tt.create_environment("Taxi-v3", render_mode="human")
    case _:
        print("Invalid choice. Defaulting to FrozenLake.")
        env = tt.create_environment("FrozenLake-v1", render_mode="human")

state, info = env.reset()
terminated = False
truncated = False

env.render()
while not terminated and not truncated:
    action = np.argmax(control.Q[state])
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)

os.system("clear")
if reward > 0:
    print("Goal reached! 🚩")
else:
    print("Agent fell in a hole. 🕳️")
env.close()
