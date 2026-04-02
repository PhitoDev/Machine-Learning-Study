import layers
import numpy as np
from activations import elu, relu, sigmoid, softmax, tanh


class Block:
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


class Dense(Block):
    def __init__(self, input_size, output_size):
        super().__init__("dense")
        self.input_size = input_size
        self.output_size = output_size
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros(output_size)

    def forward(self, X):
        """
        X: (batch_size, input_size) -> (batch_size, output_size)
        """
        self.cache["X"] = X
        Z = layers.dense(X, self.W, self.b)
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

        # Gradient w.r.t. input: dL_dZ @ W^T
        dL_dX = np.dot(dL_dZ, self.W.T)

        # Update weights and biases using SGD
        self.W -= alpha * dL_dW
        self.b -= alpha * dL_db.reshape(self.b.shape)

        return dL_dX


class ReLu(Block):
    def __init__(self) -> None:
        super().__init__("relu")

    def forward(self, X):
        self.cache["X"] = X
        return relu(X)

    def backward(self, dL_dZ, alpha):
        """
        Backpropagate through ReLU activation.
        ReLU derivative: 1 if X > 0, else 0
        """
        X = self.cache["X"]
        dL_dX = dL_dZ * (X > 0).astype(float)
        return dL_dX


class Sigmoid(Block):
    def __init__(self) -> None:
        super().__init__("sigmoid")

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z"] = sigmoid(X)
        return self.cache["Z"]

    def backward(self, dL_dZ, alpha):
        """
        Backpropagate through Sigmoid activation.
        Sigmoid derivative: sigmoid(Z) * (1 - sigmoid(Z))
        """
        Z = self.cache["Z"]
        dL_dX = dL_dZ * Z * (1 - Z)
        return dL_dX


class Tanh(Block):
    def __init__(self) -> None:
        super().__init__("tanh")

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z"] = tanh(X)
        return self.cache["Z"]

    def backward(self, dL_dZ, alpha):
        """
        Backpropagate through Tanh activation.
        Tanh derivative: 1 - tanh(Z)^2
        """
        Z = self.cache["Z"]
        dL_dX = dL_dZ * (1 - Z**2)
        return dL_dX


class Softmax(Block):
    def __init__(self) -> None:
        super().__init__("softmax")

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z"] = softmax(X)
        return self.cache["Z"]

    def backward(self, dL_dZ, alpha):
        """
        Backpropagate through Softmax activation.
        For softmax with cross-entropy loss, dL_dX = softmax(Z) - y_true
        For MSE loss: approximate derivative as softmax(Z) * (1 - softmax(Z))
        """
        Z = self.cache["Z"]
        # Element-wise product with jacobian diagonal approximation
        dL_dX = dL_dZ * Z * (1 - Z)
        return dL_dX


class ELU(Block):
    def __init__(self, alpha=1.0) -> None:
        super().__init__("elu")
        self.alpha_activation = alpha

    def forward(self, X):
        self.cache["X"] = X
        self.cache["Z"] = elu(X, self.alpha_activation)
        return self.cache["Z"]

    def backward(self, dL_dZ, alpha):
        """
        Backpropagate through ELU activation.
        ELU derivative: 1 if X > 0, else alpha * exp(X)
        """
        X = self.cache["X"]
        dL_dX = dL_dZ * np.where(X > 0, 1.0, self.alpha_activation * np.exp(X))
        return dL_dX
