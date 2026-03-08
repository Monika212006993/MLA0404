# Experiment 21 - Neural Network for Circular Data (ReLU)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Generate circular dataset
np.random.seed(0)
n = 300
X = np.random.uniform(-1, 1, (n, 2))

# Create circular class labels
y = (X[:,0]**2 + X[:,1]**2 < 0.5**2).astype(int)

# Neural Network Model
model = MLPClassifier(hidden_layer_sizes=(2,2,2),   # 3 hidden layers, 2 neurons each
                      activation='relu',
                      learning_rate_init=0.1,
                      max_iter=5000)

# Train the model
model.fit(X, y)

# Create mesh grid for decision boundary
xx, yy = np.meshgrid(np.linspace(-1,1,200),
                     np.linspace(-1,1,200))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k')

plt.title("Neural Network Classification for Circular Data (ReLU)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
