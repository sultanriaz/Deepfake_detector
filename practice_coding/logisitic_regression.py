<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=load_breast_cancer()
X=data.data
y=data.target
X_train, X_test, y_train , y_test=train_test_split(X, y , test_size=0.2, random_state=42)
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def compute_loss(y, y_hat):
    epsilon=1e-9
    return -np.mean(

        y*np.log(
            y_hat+epsilon
        )+
        (1-y)*np.log(
            1-y_hat+epsilon
        )
    )
def compute_gradients(X, y, y_hat):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    n_features = X.shape[1]
  
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
     
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
 
        loss = compute_loss(y, y_hat)
       
        dw, db = compute_gradients(X, y, y_hat)
          
        w -= lr * dw
        b -= lr * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b
weights, bias = train_logistic_regression(
    X_train, y_train, lr=0.01, epochs=1000
)
def predict(X, w, b, threshold=0.5):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= threshold).astype(int)
y_pred = predict(X_test, weights, bias)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
=======
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=load_breast_cancer()
X=data.data
y=data.target
X_train, X_test, y_train , y_test=train_test_split(X, y , test_size=0.2, random_state=42)
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def compute_loss(y, y_hat):
    epsilon=1e-9
    return -np.mean(

        y*np.log(
            y_hat+epsilon
        )+
        (1-y)*np.log(
            1-y_hat+epsilon
        )
    )
def compute_gradients(X, y, y_hat):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_hat - y))
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    n_features = X.shape[1]
  
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
     
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
 
        loss = compute_loss(y, y_hat)
       
        dw, db = compute_gradients(X, y, y_hat)
          
        w -= lr * dw
        b -= lr * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b
weights, bias = train_logistic_regression(
    X_train, y_train, lr=0.01, epochs=1000
)
def predict(X, w, b, threshold=0.5):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= threshold).astype(int)
y_pred = predict(X_test, weights, bias)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
