import pandas as pd
import numpy as np
from train_model import model_file

def read_theta(file_name: str) -> np.ndarray:
    try:
        theta = pd.read_csv(file_name).values
        theta = theta.reshape(-1, 1)
    except FileNotFoundError:
        print("File not found")
        exit()
    return theta

def read_km() -> float:
    while 42:
        try:
            km = float(input("Enter the km: "))
            return km
        except ValueError:
            print("Invalid km")

def predict(theta: np.ndarray, km: float) -> float:
    print("theta: ", theta)
    print("km: ", km)
    return theta[0] * km + theta[1]

if __name__ == "__main__":
    theta = read_theta(model_file)
    km = read_km()
    price = predict(theta, km)
    print("The price of the car is: ", price)
