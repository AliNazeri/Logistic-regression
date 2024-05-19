import numpy as np
import tensorflow as tf

# Parameters
features = 1
epochs = 10
batch_num = 1
sample_num = 100
alpha = 0.5  # Learning rate

# Initialize weights and bias
weights = np.zeros((features, 1))
bias = 0

# Generate random data
rng = np.random.default_rng()
X = rng.integers(low=0, high=10, size=(features, sample_num))
Y = rng.integers(low=0, high=2, size=(1, sample_num))

# Training loop
for epoch in range(epochs):
    for batch in range(batch_num):
        # Compute linear combination
        z = np.dot(weights.T, X) + bias
        # Apply sigmoid function
        a = tf.sigmoid(z).numpy()
        # Compute logistic loss
        loss = -np.sum(Y * np.log(a) + (1 - Y) * np.log(1 - a)) / sample_num
        print(f'Epoch {epoch + 1}, Loss: {loss}')

        # Compute gradients
        dz = a - Y
        dw = np.dot(X, dz.T) / sample_num
        db = np.sum(dz) / sample_num

    # Update weights and bias
    weights -= alpha * dw
    bias -= alpha * db

# Final cost
cost = loss / batch_num
print(f'Final Cost: {cost}')

# w1 = 0, w2 = 0, b =0
# alpha = 0.1 ## learning rate
## loop the epochs
## start one epoch
# dw1 = 0
# dw2 = 0
# db = 0
## batches loop goes here
# z = w1*x1 + w1*x2 + b
# a = sigmoid(z)
# l = -(y * math.log(a) + (1 - y) * math.log(1 - a))
# dz = a - y
# dw1 += dz * x1
# dw2 += dz * x2
# db += dz
## after loop
# j = np.sum(l)/n
# dw1 /= m
# dw2 /= m
# db /= m
## end of one epoch
# w1 = w1 - alpha * dw1
# w2 = w2 - alpha * dw2
# b = b - alpha * db
## end of epochs loop