import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Load and preprocess dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=1)

# MLP Configuration
input_size = 4
hidden_size = 4
output_size = 3
learning_rate = 0.4
momentum = 0.2
epochs = 500

# Weight Initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Momentum terms
vW1 = np.zeros_like(W1)
vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2)
vb2 = np.zeros_like(b2)

loss_history = []

# Training
for epoch in range(epochs):
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    loss = np.mean((a2 - y_train) ** 2)
    loss_history.append((epoch, loss))

    d_output = (a2 - y_train) * sigmoid_derivative(a2)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)

    dW2 = np.dot(a1.T, d_output)
    db2 = np.sum(d_output, axis=0, keepdims=True)
    dW1 = np.dot(X_train.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    vW2 = momentum * vW2 - learning_rate * dW2
    vb2 = momentum * vb2 - learning_rate * db2
    vW1 = momentum * vW1 - learning_rate * dW1
    vb1 = momentum * vb1 - learning_rate * db1

    W2 += vW2
    b2 += vb2
    W1 += vW1
    b1 += vb1

# Testing
z1_test = np.dot(X_test, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = sigmoid(z2_test)

predicted = np.argmax(a2_test, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted == actual)

# Display Epoch Losses (0, 100, 200, 300, 400)
print("\nEpoch-wise Loss (selected):")
for e, l in loss_history:
    if e % 100 == 0:
        print(f"Epoch {e}, Loss: {l:.4f}")

# Display Predictions vs Actual
print("\nPredicted:", predicted)
print("Actual:   ", actual)
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed Final Output Table for Excel-like export
df_detail = pd.DataFrame(X_test, columns=iris.feature_names)
df_detail['Target'] = actual
df_detail['Predicted'] = predicted
df_detail['Correct'] = df_detail['Target'] == df_detail['Predicted']
print("\n=== Detailed Prediction Table (all rows) ===")
print(df_detail)

# Weight Summary Table
W1_flat = W1.flatten()
W2_flat = W2.flatten()
df_weights = pd.DataFrame({
    'Layer': ['Input → Hidden'] * len(W1_flat) + ['Hidden → Output'] * len(W2_flat),
    'Weight Index': list(range(len(W1_flat))) + list(range(len(W2_flat))),
    'Weight Value': np.concatenate([W1_flat, W2_flat])
})

print("\n=== Final Weights Summary ===")
print(df_weights)  # show first few rows
