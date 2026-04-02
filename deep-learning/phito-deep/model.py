import blocks as b
import losses as l
import optimization


class Sequential:
    def __init__(self, *blocks, alpha=0.01, optimizer="sgd") -> None:
        """
        Initialize with variable number of blocks.

        Usage:
            model = Sequential(
                b.Dense(256, 128),
                b.ReLu(),
                b.Dense(128, 1),
                b.Sigmoid()
            )
        """
        self.blocks = list(blocks)
        self.alpha = alpha
        self.optimizer = optimizer

    def add(self, block) -> None:
        """Add a block to the network."""
        self.blocks.append(block)

    def setoptimizer(self, name):
        self.optimizer = name

    def train(self, X, y):
        match self.optimizer:
            case "sgd":
                losses = optimization.mini_batch_sgd(
                    model=self,
                    X=X,
                    y=y,
                    loss_class=l.MSE(),
                    alpha=self.alpha,
                )
            case _:
                raise ValueError(f"{self.optimizer} is not a valid optimizer.")

        return losses

    def forward(self, X):
        """
        Forward pass through all blocks.

        Args:
            X: input array

        Returns:
            output after passing through all blocks
        """
        output = X
        for block in self.blocks:
            output = block.forward(output)
        return output

    def backward(self, gradient):
        """
        Backward pass through all blocks.

        Args:
            gradient: dL/dY from loss function (shape: batch_size x output_size)

        Propagates gradient backwards through all layers in reverse order.
        Each block computes its parameter gradients, updates parameters,
        and returns the gradient for the previous layer.
        """
        # Start with gradient from loss and propagate backwards
        current_gradient = gradient

        # Iterate through blocks in reverse order
        for block in reversed(self.blocks):
            # Pass gradient through block and get gradient for previous layer
            current_gradient = block.backward(current_gradient, self.alpha)

    def __call__(self, X):
        """Allow model(X) syntax."""
        return self.forward(X)

    def summary(self):
        """Print model architecture."""
        print("Model Summary:")
        print("-" * 60)
        for i, block in enumerate(self.blocks):
            if isinstance(block, b.Dense):
                print(
                    f"Layer {i}: {block.name.upper():<10} | Input: {block.input_size:<5} Output: {block.output_size:<5}"
                )
            else:
                print(f"Layer {i}: {block.name.upper():<10}")
        print("-" * 60)


class SequentialBuilder:
    """Fluent API for building Sequential models."""

    def __init__(self, alpha=0.01):
        self.blocks = []
        self.alpha = alpha
        self.optimizer_name = "sgd"

    def dense(self, input_size, output_size):
        """Add a Dense layer."""
        self.blocks.append(b.Dense(input_size, output_size))
        return self

    def relu(self):
        """Add a ReLU activation."""
        self.blocks.append(b.ReLu())
        return self

    def sigmoid(self):
        """Add a Sigmoid activation."""
        self.blocks.append(b.Sigmoid())
        return self

    def tanh(self):
        """Add a Tanh activation."""
        self.blocks.append(b.Tanh())
        return self

    def softmax(self):
        """Add a Softmax activation."""
        self.blocks.append(b.Softmax())
        return self

    def elu(self, alpha_activation=1.0):
        """Add an ELU activation."""
        self.blocks.append(b.ELU(alpha_activation))
        return self

    def optimizer(self, name):
        """Set the optimizer."""
        self.optimizer_name = name
        return self

    def build(self):
        """Build and return the Sequential model."""
        return Sequential(*self.blocks, alpha=self.alpha, optimizer=self.optimizer_name)
