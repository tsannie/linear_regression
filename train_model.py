import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cost_function(X, y, theta):
    m = y.size
    return (1 / (2 * m)) * (np.sum(X.dot(theta) - y) ** 2)  # WARN X.dot(theta) = F

def grad(X, y, theta):
    m = y.size
    return (1 / m) * X.T.dot(X.dot(theta) - y)

def gradient_descent(X, y, theta, alpha, n_iters):
    theta_tmp = theta.copy()
    for i in range(n_iters):
        J = cost_function(X, y, theta_tmp)
        print("Cost function: ", J)
        theta_tmp = theta_tmp - alpha * grad(X, y, theta_tmp)
    return theta_tmp

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prediction.py <input_file>")
        exit(1)

    df = pd.read_csv(sys.argv[1])

    x = np.array(df['km']).reshape(-1, 1)
    y = np.array(df['price']).reshape(-1, 1)
    x = normalize_data(x)
    y = normalize_data(y)
    print(x.shape)
    print(y.shape)

    X = np.hstack((x, np.ones((x.shape[0], 1))))
    print(X)
    print(X.shape)

    # theta ((n + 1), 1) with random values
    theta = np.random.randn(2, 1).reshape(-1, 1)
    print(theta)

    F = X.dot(theta)

    J = cost_function(X, y, theta)
    print("Cost function: ", J)
    theta_train = gradient_descent(X, y, theta, 0.1, 100)
    print(theta_train)
    F = X.dot(theta_train)
    J = cost_function(X, y, theta_train)
    print("Cost function: ", J)

    plt.plot(x, F, 'r')
    plt.scatter(x, y)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()
