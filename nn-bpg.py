import numpy as np

# Define the sigmoid function and its derivative.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # Instead of computing derivative from x directly, we compute it using a = sigmoid(x)
    a = sigmoid(x)
    return a * (1 - a)

# ----------------------------
# Network parameters (given values)
# ----------------------------
x = 1.0  # Input

# Layer 1 parameters
W1 = 0.5
b1 = 0.1

# Layer 2 parameters
W2 = 0.8
b2 = 0.2

# Layer 3 parameters
W3 = 0.3
b3 = -0.1

# Output layer parameters
W4 = 1.2
b4 = 0.05

# Target value for regression
y_target = 1.0

# ----------------------------
# Forward Pass
# ----------------------------

# Hidden Layer 1
z1 = W1 * x + b1          # z^(1) = 0.5 * 1.0 + 0.1 = 0.6
a1 = sigmoid(z1)          # a^(1) = sigmoid(0.6)

# Hidden Layer 2
z2 = W2 * a1 + b2         # z^(2) = 0.8 * a1 + 0.2
a2 = sigmoid(z2)          # a^(2) = sigmoid(z2)

# Hidden Layer 3
z3 = W3 * a2 + b3         # z^(3) = 0.3 * a2 - 0.1
a3 = sigmoid(z3)          # a^(3) = sigmoid(z3)

# Output Layer (using identity activation for regression)
z4 = W4 * a3 + b4         # z^(4) = 1.2 * a3 + 0.05
y_pred = z4               # y = z^(4)

# Mean Squared Error Loss: L = 0.5*(y_pred - y_target)^2
loss = 0.5 * (y_pred - y_target)**2

print("=== Forward Pass ===")
print(f"Layer 1: z1 = {z1:.4f}, a1 = {a1:.4f}")
print(f"Layer 2: z2 = {z2:.4f}, a2 = {a2:.4f}")
print(f"Layer 3: z3 = {z3:.4f}, a3 = {a3:.4f}")
print(f"Output: z4 = {z4:.4f}, y_pred = {y_pred:.4f}")
print(f"Loss = {loss:.4f}")

# ----------------------------
# Backward Pass (Using the Chain Rule)
# ----------------------------

# Output Layer:
# dL/dy = y_pred - y_target
dL_dy = y_pred - y_target  # = 0.6804 - 1.0 ≈ -0.3196
# Since the output activation is identity, dL/dz4 = dL/dy.
dL_dz4 = dL_dy

# Gradients for Output Layer parameters:
dL_dW4 = dL_dz4 * a3      # dL/dW^(4) = dL/dz^(4) * a^(3)
dL_db4 = dL_dz4         # dL/db^(4) = dL/dz^(4)

print("\n=== Backward Pass ===")
print(f"dL/dz4 = {dL_dz4:.4f}")
print(f"dL/dW4 = {dL_dW4:.4f}, dL/db4 = {dL_db4:.4f}")

# Hidden Layer 3:
# dL/dz3 = dL/dz4 * (d(z4)/da3) * (da3/dz3)
# d(z4)/da3 = W4, and da3/dz3 = a3*(1-a3)
dL_dz3 = dL_dz4 * W4 * (a3 * (1 - a3))
dL_dW3 = dL_dz3 * a2    # dL/dW^(3) = dL/dz^(3) * a^(2)
dL_db3 = dL_dz3         # dL/db^(3) = dL/dz^(3)

print(f"dL/dz3 = {dL_dz3:.4f}")
print(f"dL/dW3 = {dL_dW3:.4f}, dL/db3 = {dL_db3:.4f}")

# Hidden Layer 2:
# dL/dz2 = dL/dz3 * (d(z3)/da2) * (da2/dz2)
# d(z3)/da2 = W3, and da2/dz2 = a2*(1-a2)
dL_dz2 = dL_dz3 * W3 * (a2 * (1 - a2))
dL_dW2 = dL_dz2 * a1    # dL/dW^(2) = dL/dz^(2) * a^(1)
dL_db2 = dL_dz2         # dL/db^(2) = dL/dz^(2)

print(f"dL/dz2 = {dL_dz2:.4f}")
print(f"dL/dW2 = {dL_dW2:.4f}, dL/db2 = {dL_db2:.4f}")

# Hidden Layer 1:
# dL/dz1 = dL/dz2 * (d(z2)/da1) * (da1/dz1)
# d(z2)/da1 = W2, and da1/dz1 = a1*(1-a1)
dL_dz1 = dL_dz2 * W2 * (a1 * (1 - a1))
dL_dW1 = dL_dz1 * x     # dL/dW^(1) = dL/dz^(1) * x  (here x = 1.0)
dL_db1 = dL_dz1         # dL/db^(1) = dL/dz^(1)

print(f"dL/dz1 = {dL_dz1:.4f}")
print(f"dL/dW1 = {dL_dW1:.4f}, dL/db1 = {dL_db1:.4f}")
