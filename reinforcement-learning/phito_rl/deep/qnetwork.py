import numpy as np
from phitodeep.loss import MeanSquaredError
from phitodeep.model import SequentialBuilder


class QNetwork:
    def __init__(self, input_size, output_size) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.model = (
            SequentialBuilder()
            .dense(input_size, 128)
            .relu()
            .dense(128, output_size)
            .loss(MeanSquaredError())
            .optimizer("adam")
            .alpha(0.001)
            .epochs(150)
            .batch(64)
            .build()
        )

    def predict(self, X):
        return self.model.predict(X)

    def copy(self):
        new_network = QNetwork(self.input_size, self.output_size)
        new_network.model = self.model.copy()
        return new_network


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.rng = np.random.default_rng(42)
        self.buffer = []
        self.buffer_size = buffer_size
        self.pointer = 0

    def push(self, s, a, r, s_prime, done):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s_prime, done))
        self.pointer = (self.pointer + 1) % self.buffer_size

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        input_size,
        output_size,
        capacity,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
    ):
        self.rng = np.random.default_rng(67)
        self.input_size = input_size
        self.output_size = output_size
        self.q_network = QNetwork(input_size, output_size)
        self.target_network = self.q_network.copy()
        self.replay_buffer = ReplayBuffer(capacity)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, training=True):
        if training and self.rng.uniform() < self.epsilon:
            return self.rng.integers(0, self.output_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        q_values = self.q_network.predict(states)
        next_q_values = self.target_network.predict(next_states)
        targets = q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(
                    next_q_values[i]
                )
        split = int(0.8 * len(targets))
        train_states, train_targets = states[:split], targets[:split]
        test_states, test_targets = states[split:], targets[split:]
        self.q_network.model.train(
            train_states, train_targets, test_states, test_targets
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network = self.q_network.copy()
