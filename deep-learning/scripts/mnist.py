"""
Simple test to verify the neural network implementation works correctly.
"""

import numpy as np
from datasets import load_dataset
from phitodeep.loss import CategoricalCrossEntropy
from phitodeep.model import SequentialBuilder

train_dataset = load_dataset("ylecun/mnist", split="train")
test_dataset = load_dataset("ylecun/mnist", split="test")

X_train = train_dataset["image"]
y_train = train_dataset["label"]
X_test = test_dataset["image"]
y_test = test_dataset["label"]

X_train = np.array(X_train).astype(np.float32) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test).astype(np.float32) / 255.0
y_test = np.array(y_test)

print("=" * 70)
print("Testing Phito-Deep Neural Network Framework")
print("=" * 70)
print("Data Shape: ")
print(X_train.shape, y_train.shape + "\n")

model = (
    SequentialBuilder()
    .flatten()
    .dense(784, 128)
    .relu()
    .dense(128, 10)
    .softmax()
    .optimizer("adam")
    .loss(CategoricalCrossEntropy())
    .alpha(0.001)
    .epochs(100)
    .batch(64)
    .build()
)

model.summary()

y_pred = model(X_test)
accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test) * 100
print(f"Test Accuracy before training: {accuracy:.4f} %")

print("=" * 70)
print("Starting Training")
print("=" * 70)

model.train(X_train, y_train, X_test, y_test)

y_pred = model(X_test)
accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test) * 100
print(f"Test Accuracy after training: {accuracy:.4f} %")
