# linear_regression

## Description

This project provides a simple implementation of [linear regression](https://en.wikipedia.org/wiki/Linear_regression) using the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm in Python.

- `train_model.py` takes a [CSV](https://en.wikipedia.org/wiki/) file as input and trains the model using the gradient descent algorithm. After that the model is saved.
- `predict.py` retrieves the trained model and makes predictions based on the given features.

## Installation

```bash
git clone git@github.com:tsannie/linear_regression.git && cd linear_regression
pip install -r requirements.txt
```

## Usage

```
usage: train_model.py [-h] [-f 'file_name'.csv] [-t 'column_name']
                      [-n N_ITERS] [-r LEARNING_RATE] [-g]

optional arguments:
  -h, --help            show this help message and exit
  -f 'file_name'.csv, --file 'file_name'.csv
                        File name csv (default: data.csv)
  -t 'column_name', --target 'column_name'
                        Target column (default: price)
  -n N_ITERS, --n_iters N_ITERS
                        Number of iterations (default: 100)
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate (alpha) (default: 1)
  -g, --graph           Show graph
```
