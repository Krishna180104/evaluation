import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv("bank_loan_approval_dataset.csv")
X = df.drop(columns=["LoanApproved"])  
y = df["LoanApproved"]  

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)  
    predictions = sigmoid(np.dot(X, weights))  
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))  # Log loss
    return cost

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape  
    weights = np.zeros(n)  
    cost_history = []  

    for i in range(epochs):
        predictions = sigmoid(np.dot(X, weights))  
        gradient = (1/m) * np.dot(X.T, (predictions - y))  
        weights -= learning_rate * gradient  # Update weights using learning rate
        cost = compute_cost(X, y, weights)  # Compute new cost
        cost_history.append(cost)  # Store cost for monitoring progress

    return weights, cost_history



X_bias = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones

# Convert y to a NumPy array
y = y.values

# Train the model
weights, cost_history = gradient_descent(X_bias, y, learning_rate=0.1, epochs=5000)

print("Final Weights:", weights)


def predict(X, weights, threshold=0.5):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= threshold).astype(int)

# Make predictions
y_pred = predict(X_bias, weights)

# Evaluate accuracy
accuracy = np.mean(y_pred == y) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
