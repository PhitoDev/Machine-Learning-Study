import numpy as np


class Layer:
    def __init__(self, name) -> None:
        self.name = name
        self.cache = {}

    def forward(self, X):
        raise NotImplementedError(f"Block '{self.name}' must implement forward method")

    def backward(self, dL_dZ, alpha):
        """
        Backward pass through the block.

        Args:
            dL_dZ: gradient of loss w.r.t. output of this block
            alpha: learning rate

        Returns:
            dL_dX: gradient of loss w.r.t. input (to pass to previous layer)
        """
        raise NotImplementedError(f"Block '{self.name}' must implement backward method")


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__("dense")
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, X):
        """
        X: (batch_size, input_size) -> (batch_size, output_size)
        """
        self.cache["X"] = X
        Z = np.dot(X, self.W) + self.b
        return Z

    def backward(self, dL_dZ, alpha):
        """
        Backpropagate through Dense layer.

        Args:
            dL_dZ: (batch_size, output_size) - gradient of loss w.r.t. output
            alpha: learning rate

        Returns:
            dL_dX: (batch_size, input_size) - gradient to pass to previous layer
        """
        X = self.cache["X"]
        m = X.shape[0]  # batch size

        # Gradient w.r.t. weights: (1/m) * X^T @ dL_dZ
        dL_dW = np.dot(X.T, dL_dZ) / m

        # Gradient w.r.t. bias: (1/m) * sum(dL_dZ)
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True) / m

        # Update weights and biases using SGD
        self.W -= alpha * dL_dW
        self.b -= alpha * dL_db.reshape(self.b.shape)

        # Gradient w.r.t. input: dL_dZ @ W^T
        dL_dX = np.dot(dL_dZ, self.W.T)

        return dL_dX
