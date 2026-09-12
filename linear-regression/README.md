# Linear Regression from Scratch

A small educational project that implements **linear regression using gradient descent from scratch**, without relying on machine learning libraries like scikit-learn. The goal is to understand how a linear model learns its parameters step by step.

## Overview

The notebook builds a simple linear model that predicts the **price of a house** based on the **number of rooms**, using a small synthetic dataset:

```python
features = [1, 2, 3, 5, 6, 7]        # number of rooms
labels   = [155, 197, 244, 356, 407, 448]  # price
```

The model learns two parameters:
- **base_price** — the intercept (price when rooms = 0)
- **price_per_room** — the slope (price added per extra room)

## What's Implemented

### 1. Visualization Helpers
- `line()` — draws a line given a slope and intercept.
- `data_points()` — scatters the dataset points on the plot.

### 2. Learning Algorithms ("Tricks")
Three different update rules for adjusting the line toward the data, presented in increasing order of sophistication:

- **Simple Trick** — nudges the parameters by small random amounts based on whether the prediction was too high or too low.
- **Square Trick** — updates parameters proportionally to the squared error (equivalent to a single step of gradient descent for squared loss).
- **Absolute Trick** — updates parameters by a fixed learning-rate step in the direction of the error (based on absolute error).

### 3. Training Loop
`linear_regression()` runs the chosen trick (square trick by default) for a number of epochs, picking a random data point each iteration, and plots the evolving line(s) against the data points.

### 4. Error Evaluation
`rmse()` — computes the **Root Mean Squared Error** between true labels and predictions, used to track how well the model fits the data over training.

### 5. Training with Error Tracking
An extended version of `linear_regression()` that:
- Records RMSE at every epoch
- Plots two charts side by side:
  1. The final fitted line against the data points
  2. RMSE over epochs, showing convergence

## Requirements

```
numpy
matplotlib
```

## How to Run

Open the notebook in Jupyter and run all cells in order:

```bash
jupyter notebook linear_regression.ipynb
```

## Example Output

After training, the model prints the learned parameters, e.g.:

```
Price per room: 50.66
Base price: 99.80
```

## Notes

- This project is meant for learning purposes — it demonstrates the intuition behind gradient descent (via the "tricks") rather than being a production-ready ML tool.
- The dataset is intentionally tiny and synthetic to make the learning process easy to visualize.
