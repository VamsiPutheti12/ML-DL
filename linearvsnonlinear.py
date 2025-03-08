import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate a linear dataset
np.random.seed(42)
x_linear = np.linspace(0, 10, 20).reshape(-1, 1)
y_linear = 2 * x_linear + 5 + np.random.normal(0, 2, x_linear.shape)

# Train a linear regression model
linear_model = LinearRegression()
linear_model.fit(x_linear, y_linear)
y_linear_pred = linear_model.predict(x_linear)

# Generate a nonlinear dataset
x_nonlinear = np.linspace(0, 10, 100).reshape(-1, 1)
y_nonlinear = np.sin(x_nonlinear) * 10 + np.random.normal(0, 2, x_nonlinear.shape)

# Train a linear model on nonlinear data
linear_model.fit(x_nonlinear, y_nonlinear)
y_nonlinear_pred = linear_model.predict(x_nonlinear)

# Train a polynomial model (degree=3) for better fit
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(x_nonlinear, y_nonlinear)
y_poly_pred = poly_model.predict(x_nonlinear)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear Data
axes[0].scatter(x_linear, y_linear, label="Data", color="blue")
axes[0].plot(x_linear, y_linear_pred, label="Linear Regression", color="red")
axes[0].set_title("Linear Regression on Linear Data")
axes[0].legend()

# Nonlinear Data
axes[1].scatter(x_nonlinear, y_nonlinear, label="Data", color="blue")
axes[1].plot(x_nonlinear, y_nonlinear_pred, label="Linear Regression (Fails)", color="red")
axes[1].plot(x_nonlinear, y_poly_pred, label="Polynomial Regression (Better Fit)", color="green")
axes[1].set_title("Linear vs Polynomial Regression on Nonlinear Data")
axes[1].legend()

plt.show()