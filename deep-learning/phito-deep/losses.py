import numpy as np


class LossBase:
    def __init__(self, name) -> None:
        self.name = name

    def loss_func(self, y_pred, y_true):
        raise NotImplementedError(f"{self.name} must implement the loss_func method.")

    def loss_gradient(self, y_pred, y_true):
        raise NotImplementedError(
            f"{self.name} must implement the loss_gradient method."
        )


class MSE(LossBase):
    def __init__(self) -> None:
        super().__init__("MSE")

    def loss_func(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def loss_gradient(self, y_pred, y_true):
        m = len(y_true)
        return 2 * (y_pred - y_true) / m
