# Experiment 19 - Neural Network for Circular Data Class

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Generate circular dataset
np.random.seed(0)
n = 300
X = np.random.uniform(-1, 1, (n, 2))

# Circular boundary
y = (X[:,0]**2 + X[:,1]**2 < 0.5**2).astype(int)

# Neural Network Model
model = MLPClassifier(hidden_layer_sizes=(3,3),   # 2 hidden layers, 3 neurons
                      activation='identity',      # Linear activation
                      learning_rate_init=0.03,
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
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')

plt.title("Neural Network Classification for Circular Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
