import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Generate sample non-linear data (quadratic relationship)
np.random.seed(42)
X = 10 * np.random.rand(100, 1) - 5  # 100 values between -5 and 5
y = 2 + 3 * X + 1.5 * X**2 + np.random.randn(100, 1) * 5  # Quadratic relationship with noise

# Transform features into polynomial features (degree = 2 for quadratic)
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)  # This creates features: [X, X^2]

# Add a bias column to X_poly
X_poly_b = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]

# Compute the optimal parameters using the Normal Equation:
# theta = (X_poly_b^T * X_poly_b)^(-1) * X_poly_b^T * y
theta = np.linalg.inv(X_poly_b.T.dot(X_poly_b)).dot(X_poly_b.T).dot(y)
print("Optimized parameters (theta):")
print(theta)

# Generate new input data for predictions
X_new = np.linspace(-5, 5, 100).reshape(-1, 1)
X_new_poly = poly.transform(X_new)
X_new_poly_b = np.c_[np.ones((X_new_poly.shape[0], 1)), X_new_poly]
y_predict = X_new_poly_b.dot(theta)

# Plot the original data and the polynomial regression curve
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_new, y_predict, color="red", linewidth=2, label="Polynomial Regression (OLS)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression using the Normal Equation")
plt.show()
