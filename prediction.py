import pandas as pd
import numpy as np
from train_model import model_file

def read_theta(file_name: str) -> np.ndarray:
    try:
        df = pd.read_csv(file_name, header=None).values
        theta = df[1].astype(float)
        header = df[0]
    except FileNotFoundError:
        print("File not found")
        exit()
    return theta, header

def read_ft(feature_name: str) -> float:
    while 42:
        try:
            v = float(input("Enter the {} value: ".format(feature_name)))
            return v
        except ValueError:
            print("Invalid value")

def predict(theta: np.ndarray, v: float) -> float:
    return theta[1] + (theta[0] * v)

if __name__ == "__main__":
    theta, header = read_theta(model_file)
    v = read_ft(header[1])
    p = predict(theta, v)
    print("The predicted {} is: {}".format(header[0], p))
