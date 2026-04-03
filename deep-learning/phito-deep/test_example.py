"""
Simple test to verify the neural network implementation works correctly.
"""

import numpy as np
from loss import BinaryCrossEntropy
from model import SequentialBuilder

# Set random seed for reproducibility
np.random.seed(42)

# Generate simple synthetic data
n_samples = 100
X_train = np.random.randn(n_samples, 10)  # 100 samples, 10 features
y_train = np.random.randint(0, 2, (n_samples, 1)).astype(float)  # Binary labels

print("=" * 70)
print("Testing Phito-Deep Neural Network Framework")
print("=" * 70)

# Build model using SequentialBuilder (fluent API)
print("\nBuilding model: 10 -> 16 -> 8 -> 1")
model = (
    SequentialBuilder()
    .dense(10, 16)
    .relu()
    .dense(16, 8)
    .relu()
    .dense(8, 1)
    .sigmoid()
    .optimizer("sgd")
    .loss(BinaryCrossEntropy())
    .batch(32)
    .alpha(0.1)
    .epochs(600)
    .build()
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Forward pass before training
print("\n" + "-" * 70)
print("Forward pass test (before training):")
print("-" * 70)
y_pred_before = model(X_train[:5]).round()
print(f"Predictions shape: {y_pred_before.shape}")
print(f"Sample predictions (first 5):\n{y_pred_before.flatten()}")
print(f"Sample true labels (first 5):\n{y_train[:5].flatten()}")

# Train the model (uses default epochs=1000, batch_size=1)
print("\n" + "=" * 70)
print("Training model (1000 epochs with default settings)...")
print("=" * 70)
losses = model.train(X_train, y_train)

# Forward pass after training
print("\n" + "-" * 70)
print("Predictions after training:")
print("-" * 70)
y_pred_after = model(X_train[:10]).round()
print(f"Predictions shape: {y_pred_after.shape}")
print(f"Sample predictions (first 10):\n{y_pred_after.flatten()}")
print(f"Sample true labels (first 10):\n{y_train[:10].flatten()}")

# Print loss statistics
print(f"\nInitial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Loss improvement: {(losses[0] - losses[-1]):.6f}")

print("\n" + "=" * 70)
print("✓ Test completed successfully!")
print("=" * 70)
