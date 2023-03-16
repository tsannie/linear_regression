import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize_data, gradient, denormalize_theta, cost_function, r_squared, mae, mape, mse

# Default values
file_name = "data.csv"
learning_rate = 1
n_iters = 100
target = "price"

# const
nb_animation = 100
model_file = "theta.csv"

class LinearRegression:
    """Linear Regression model"""
    def __init__(self, learning_rate: float, n_iters: int, file_name: str, target_name: str) -> None:
        try:
            df = pd.read_csv(file_name)
        except FileNotFoundError:
            exit("File not found")
        if df.shape[1] != 2 or df.shape[0] < 2:
            exit("Invalid file format")
        if target not in df.columns.values:
            exit("Invalid target column")
        if learning_rate <= 0 or n_iters <= 0:
            exit("Invalid learning rate or number of iterations")

        self.history_cost = np.zeros(n_iters)
        self.target_name = target_name
        self.feature_name = df.columns.values[df.columns.values != target_name][0]

        self.target = np.array(df[self.target_name]).reshape(-1, 1)
        self.feature = np.array(df[self.feature_name]).reshape(-1, 1)
        self.x = normalize_data(self.feature)
        self.y = normalize_data(self.target)

        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def train(self, graph: bool = False) -> None:
        """Train the model using gradient descent"""
        theta = np.random.randn(2, 1).reshape(-1, 1)
        X = np.hstack((self.x, np.ones((self.x.shape[0], 1))))
        X_denormalized = np.hstack((self.feature, np.ones((self.feature.shape[0], 1))))

        for i in range(self.n_iters):
            theta = theta - self.learning_rate * gradient(X, self.y, theta)
            if graph:
                self.history_cost[i] = cost_function(X, self.y, theta)
                if self.n_iters < nb_animation or not i % (self.n_iters // nb_animation):
                    tmp_theta = denormalize_theta(theta, self.target, self.feature)
                    tmp_model = X_denormalized.dot(tmp_theta)
                    self.graph_animation(tmp_model)

        plt.close()
        self.theta = denormalize_theta(theta, self.target, self.feature)
        self.model = X_denormalized.dot(self.theta)

    def dump_model(self, file_name: str) -> bool:
        """Dump the model to a file"""
        df = pd.DataFrame(self.theta.T, columns=[self.target_name, self.feature_name])
        try:
            df.to_csv(file_name, index=False)
            return True
        except FileNotFoundError:
            print("Error: Impossible to save.")
            return False

    def graph_animation(self, model: np.ndarray) -> None:
        """Animation graphic with pyplot"""
        plt.plot(self.feature, model, 'r')
        plt.scatter(self.feature, self.target)
        plt.xlabel(self.feature_name)
        plt.ylabel(self.target_name)
        plt.title("Gradient descent in progress")
        plt.legend(['model', 'data'])
        plt.draw()
        plt.pause(1e-2)
        plt.clf()

    def graph(self) -> None:
        """Final graphic with pyplot"""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Linear regression graph
        ax1.set_title("Linear regression with R-squared: {:.4f}".format(r_squared(self.target, self.model)))
        ax1.plot(self.feature, self.model, 'g')
        ax1.scatter(self.feature, self.target)
        ax1.set_xlabel(self.feature_name)
        ax1.set_ylabel(self.target_name)
        ax1.legend(['model', 'data'])

        # Cost History graph
        ax2.set_title("Cost history")
        ax2.plot(range(self.n_iters), self.history_cost)
        ax2.set_xlabel("Number of iterations")
        ax2.set_ylabel("Cost")

        plt.show()

    def evaluate(self) -> dict:
        """Evaluate the model"""
        return {
            "R-squared (R^2)": r_squared(self.target, self.model),
            "Mean absolute error (MAE)": mae(self.target, self.model),
            "Mean absolute % error (MAPE)": mape(self.target, self.model),
            "Mean squared error (MSE)": mse(self.target, self.model)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File name csv (default: data.csv)", default=file_name, type=str, metavar="\'file_name\'.csv")
    parser.add_argument("-t", "--target", help="Target column (default: price)", default=target, type=str, metavar="\'column_name\'")
    parser.add_argument("-n", "--n_iters", help="Number of iterations (default: 100)", default=n_iters, type=int)
    parser.add_argument("-r", "--learning_rate", help="Learning rate (alpha) (default: 1)", default=learning_rate, type=float)
    parser.add_argument("-g", "--graph", help="Show graph", default=False, action="store_true")
    args = parser.parse_args()

    lr = LinearRegression(args.learning_rate, args.n_iters, args.file, args.target)
    lr.train(args.graph)
    lr.dump_model(model_file)
    for key, value in lr.evaluate().items():
        print("{:<30} -> {:.4f}".format(key, value))
    if args.graph:
        lr.graph()

