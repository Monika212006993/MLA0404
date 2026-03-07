import numpy as np

# Input data (Two-class problem)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Target output
y = np.array([[0],
              [1],
              [1],
              [0]])

# Learning rate
lr = 0.03

# Random weights
np.random.seed(0)

W1 = np.random.randn(2,3)
W2 = np.random.randn(3,3)
W3 = np.random.randn(3,1)

# Linear activation
def linear(x):
    return x

# Training
for epoch in range(1000):

    # Forward propagation
    h1 = linear(np.dot(X, W1))
    h2 = linear(np.dot(h1, W2))
    output = linear(np.dot(h2, W3))

    # Error
    error = y - output

    # Backpropagation (simple gradient update)
    dW3 = np.dot(h2.T, error)
    dW2 = np.dot(h1.T, np.dot(error, W3.T))
    dW1 = np.dot(X.T, np.dot(np.dot(error, W3.T), W2.T))

    # Update weights
    W3 += lr * dW3
    W2 += lr * dW2
    W1 += lr * dW1

print("Predicted Output:")
print(output)
