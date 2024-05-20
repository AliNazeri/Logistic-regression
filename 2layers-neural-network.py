import numpy as np
import tensorflow as tf

# Parameters
features = 2
hidden_layer = 3
outputs_num = 1
epochs = 100
batch_num = 1
sample_num = 100 # => m
alpha = 0.5  # Learning rate

rng = np.random.default_rng()
# Initialize weights and bias
weights = rng.integers(low=0, high=11, size=(hidden_layer, features)) * 0.01
weights2 = rng.integers(low=0, high=11, size=(outputs_num, hidden_layer)) * 0.01
bias = np.zeros((3,1))
bias2 = 0

# Generate random data
X = rng.integers(low=0, high=21, size=(features, sample_num))
Y = rng.integers(low=0, high=2, size=(1, sample_num))

# Training loop
for epoch in range(epochs):
    for batch in range(batch_num):
        # First layer
        z = np.dot(weights, X) + bias
        a = np.array(tf.tanh(z))

        z2 = np.dot(weights2, a) + bias2
        a2 = np.array(tf.sigmoid(z2))

        # Compute logistic loss
        loss = -np.sum(Y * np.log(a2) + (1 - Y) * np.log(1 - a2)) / sample_num
        print(f'Epoch {epoch + 1}, Loss: {loss}')

        # Compute gradients
        dz2 = a2 - Y

        dw2 = np.dot(dz2, a.T) / sample_num
        db2 = np.sum(dz2,axis=1,keepdims=True) / sample_num

        da = np.dot(weights2.T, dz2)
        tanh_derivation = (1 - np.tanh(z) ** 2)

        dz = np.multiply(da, tanh_derivation)
        dw = np.dot(dz, X.T) / sample_num
        db = np.sum(dz,axis=1,keepdims=True) / sample_num

    # Update weights and bias
    weights -= alpha * dw
    bias -= alpha * db
    weights2 -= alpha * dw2
    bias2 -= alpha * db2

# Final cost
cost = loss / batch_num
print(f'Final Cost: {cost}')