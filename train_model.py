import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = "data.csv"
model_file = "theta.csv"
learning_rate = 1
n_iters = 100

def dump_model(theta: np.ndarray, file_name: str) -> None:
    df = pd.DataFrame(theta.T, columns=['a', 'b'])
    try:
        df.to_csv(file_name, index=False)
    except FileNotFoundError:
        print("impossible to save the model")

def model(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X.dot(theta)

def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = y.size
    F = model(theta, X)
    return (1 / (2 * m)) * (np.sum(F - y) ** 2)

def grad(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    m = y.size
    F = model(theta, X)
    return (1 / m) * X.T.dot(F - y)

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, n_iters: int) -> np.ndarray:
    for i in range(n_iters):
        theta = theta - alpha * grad(X, y, theta)
    return theta

def stats(model: np.ndarray, y: np.ndarray) -> float:
    u = ((y - model) ** 2 ).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v

def denormalize_theta(theta: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    gap_y = np.max(y) - np.min(y)
    gap_x = np.max(x) - np.min(x)
    a = theta[0] * gap_y / gap_x
    b = theta[1] * gap_y + np.min(y) - a * np.min(x)
    return np.array([a, b]).reshape(-1, 1)

def normalize_data(X: np.ndarray) -> np.ndarray:
    return (X - np.min(X)) / (np.max(X) - np.min(X))

if __name__ == "__main__":
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print("File not found")
        exit()

    # Prepare the data with price as target and km as feature
    x_og = np.array(df['km']).reshape(-1, 1)
    y_og = np.array(df['price']).reshape(-1, 1)
    x = normalize_data(x_og)
    y = normalize_data(y_og)

    # Initialize the parameters of the model (theta = [a, b])
    X = np.hstack((x, np.ones((x.shape[0], 1))))
    theta = np.random.randn(2, 1).reshape(-1, 1)

    # Train the model using gradient descent
    theta = gradient_descent(X, y, theta, learning_rate, n_iters)

    # Dump the model to a file
    theta_dump = denormalize_theta(theta, y_og, x_og)

    dump_model(theta_dump, model_file)

    # Final trained model
    F = model(theta, X)

     # Print the R-squared value of the trained model
    print("Performance: {:.4f}".format(stats(F, y)) )

    # Graphic with pyplot
    plt.plot(x, F, 'r')
    plt.scatter(x, y)
    plt.xlabel('km')
    plt.ylabel('price')
    plt.show()
