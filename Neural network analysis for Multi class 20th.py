# Experiment 20 - Neural Network for Multi Class Data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

# Generate multi-class dataset
X, y = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=1.5)

# Neural Network Model
model = MLPClassifier(hidden_layer_sizes=(2,2),   # 2 hidden layers, 2 neurons
                      activation='identity',      # Linear activation
                      learning_rate_init=0.01,
                      max_iter=5000)

# Train the model
model.fit(X, y)

# Create mesh grid for decision boundary
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolors='k')

plt.title("Neural Network Multi-Class Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
