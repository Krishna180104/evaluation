import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("bank_loan_approval_dataset.csv")
X = df.drop(columns=["LoanApproved"]).values
y = df["LoanApproved"].values.reshape(-1, 1)

# Normalize input features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Add bias term (extra column of 1s) for easy computation
X = np.c_[np.ones(X.shape[0]), X]

# Initialize weights randomly
np.random.seed(42)
weights = np.random.randn(X.shape[1], 1) * 0.01  

# Perceptron activation function (Step function)
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Train using Perceptron Learning Rule
def train_perceptron(X, y, weights, learning_rate=0.01, epochs=1000):
    m = X.shape[0]  # Number of training examples

    for epoch in range(epochs):
        for i in range(m):
            z = np.dot(X[i], weights)  
            y_pred = step_function(z)  
            error = y[i] - y_pred  
            weights += learning_rate * error * X[i].reshape(-1, 1)  

    return weights

# Train the perceptron
weights = train_perceptron(X, y, weights, learning_rate=0.1, epochs=1000)

# Prediction function
def predict(X, weights):
    z = np.dot(X, weights)
    return step_function(z)

# Make predictions
y_pred = predict(X, weights)

# Evaluate accuracy
accuracy = np.mean(y_pred == y) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
