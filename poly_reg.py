import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample non-linear data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3  # Values between -3 and 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)  # Quadratic function with noise

# Transform the features to include polynomial terms
degree = 2  # Choose the degree of polynomial
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Train the model
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Predictions
X_new = np.linspace(-3, 3, 100).reshape(100, 1)  # New inputs
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

# Plot results
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_new, y_new, color="red", linewidth=2, label=f"Polynomial Degree {degree}")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression Example")
plt.show()