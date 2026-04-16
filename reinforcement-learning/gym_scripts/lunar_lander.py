import os
import sys
import time

import gymnasium as gym

# --- FOOLPROOF PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from phito_rl.deep.qnetwork import DQNAgent

env = gym.make("LunarLander-v3", render_mode="ansi")

episodes = 1000
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
capacity = int(0.8 * episodes)
gamma = 0.99
epsilon = 1.0
batch_size = int(0.4 * capacity)

agent = DQNAgent(input_size, output_size, capacity, gamma, epsilon)

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action = agent.select_action(state, training=True)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
        state = next_state

env.close()

for episode in range(episodes):
    agent.train(batch_size)
    os.system("clear")
    if episode % 100 == 0:
        agent.update_target_network()
        print(f"Episode {episode}, epsilon: {agent.epsilon:.2f}")

print("Training complete.")

env = gym.make("LunarLander-v3", render_mode="human")
state, _ = env.reset()
done = False
truncated = False
while not done and not truncated:
    action = agent.select_action(state, training=False)
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state

time.sleep(0.01)
env.close()

if reward >= 100:
    print("Success!")
else:
    print("Failure!")
