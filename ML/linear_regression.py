import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (Linear Relationship)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# Add bias term (X0 = 1 for all)
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 to each instance

# Initialize parameters (weights)
theta = np.random.randn(2, 1)  # Random initialization

# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

# Gradient Descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print("Optimized Theta (Parameters):")
print(theta)

# Predictions
X_new = np.array([[0], [2]])  # New input values
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add bias term
y_predict = X_new_b.dot(theta)

# Plot Data and Regression Line
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_new, y_predict, color="red", label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression using Gradient Descent")
plt.show()
