import numpy as np


def mini_batch_sgd(model, X, y, loss_class, alpha=0.01, epochs=1000, batch_size=1):
    losses = []

    for epoch in range(epochs):
        for _ in range(len(X) // batch_size):
            indices = np.random.randint(0, len(X), batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # Forward pass
            y_pred = model.forward(X_batch)
            loss = loss_class.loss_func(y_pred, y_batch)

            # Compute gradient of loss w.r.t. predictions (for MSE)
            # dL/dy = 2 * (y_pred - y_true) / m
            dy = loss_class.loss_gradient(y_pred, y_batch)

            # Backward pass with gradient of loss
            model.backward(dy)

        y_pred = model.forward(X)
        loss = loss_class.loss_func(y_pred, y)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return losses
