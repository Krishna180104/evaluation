import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv("bank_loan_approval_dataset.csv")
x = df.drop(columns=["LoanApproved"]).values
y = df["LoanApproved"].values.reshape(-1,1)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

input_size = x.shape[1]
hidden_size = 8
output_size = 1

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01  
b1 = np.zeros((1, hidden_size))  
W2 = np.random.randn(hidden_size, output_size) * 0.01  
b2 = np.zeros((1, output_size))  

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1  
    A1 = sigmoid(Z1)  
    Z2 = np.dot(A1, W2) + b2  
    A2 = sigmoid(Z2)  
    return Z1, A1, Z2, A2  

def compute_cost(A2, y):
    m = y.shape[0]
    return (-1/m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))

def backpropagation(X, y, Z1, A1, A2, W2):
    m = y.shape[0]

    dZ2 = A2 - y  
    dW2 = (1/m) * np.dot(A1.T, dZ2)  
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)  

    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)  
    dW1 = (1/m) * np.dot(X.T, dZ1)  
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)  

    return dW1, db1, dW2, db2

def train(X, y, W1, b1, W2, b2, learning_rate=0.1, epochs=5000):
    cost_history = []

    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(A2, y)
        dW1, db1, dW2, db2 = backpropagation(X, y, Z1, A1, A2, W2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        cost_history.append(cost)
    
    return W1, b1, W2, b2, cost_history

W1, b1, W2, b2, cost_history = train(x, y, W1, b1, W2, b2, learning_rate=0.1, epochs=5000)

def predict(X, W1, b1, W2, b2, threshold=0.5):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return (A2 >= threshold).astype(int)

y_pred = predict(x, W1, b1, W2, b2)
accuracy = np.mean(y_pred == y) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
