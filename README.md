# ML-DL
Machine &amp; Deep Learning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate moon-shaped dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear model (Logistic Regression)
linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

# Train a neural network with non-linearity
nn_model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(linear_model, X, y, axes[0], "Linear Model (Fails)")
plot_decision_boundary(nn_model, X, y, axes[1], "Neural Network (Works!)")
plt.show()
