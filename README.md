# linear_regression

![screen](https://i.imgur.com/aMdslTp.gif)

## Description

This project provides a simple implementation of [linear regression](https://en.wikipedia.org/wiki/Linear_regression) using the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) algorithm in Python.

- `train_model.py` takes a [CSV](https://en.wikipedia.org/wiki/) file as input and trains the model using the gradient descent algorithm. After that the model is saved.
- `predict.py` retrieves the trained model and makes predictions based on the given features.

After running `train_model.py`, performance [metrics](https://www.qualdo.ai/blog/complete-list-of-performance-metrics-for-monitoring-regression-models/) such as R-squared, Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Mean Squared Error (MSE) will be calculated.

## Installation

```bash
git clone git@github.com:tsannie/linear_regression.git && cd linear_regression
pip install -r requirements.txt
```

## Usage

```
python train_model.py [-h] [-f 'file_name'.csv] [-t 'column_name']
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

```
python predict.py
```

## Author

- [@tsannie](https://github.com/tsannie)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
