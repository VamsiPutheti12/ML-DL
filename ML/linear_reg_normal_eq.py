import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)         # 100 random samples for the feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship: y = 4 + 3*X, plus some noise

# Add bias term (column of ones) to include the intercept in the model
X_b = np.c_[np.ones((100, 1)), X]        # Now X_b has two columns: [1, X]

# Compute the optimal parameters using the Normal Equation
# theta = (X_b.T * X_b)^(-1) * X_b.T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Optimized parameters (theta):")
print(theta_best)

# Make predictions on new data points
X_new = np.array([[0], [2]])            # New input values for prediction
X_new_b = np.c_[np.ones((2, 1)), X_new]   # Add bias term
y_predict = X_new_b.dot(theta_best)       # Predicted outputs
print("Predictions for new values:")
print(y_predict)

# Plot the data and the regression line
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X_new, y_predict, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression using the Normal Equation")
plt.show()
