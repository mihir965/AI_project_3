import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Generate sample data
np.random.seed(0)
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = X.flatten()**2 + np.random.normal(0, 10, X.shape[0])  # Adding some noise

# Fit Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Fit Neural Network Regressor
# Using a simple MLP with one hidden layer
nn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=10000, random_state=1)
nn_model.fit(X, y)
y_nn_pred = nn_model.predict(X)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='lightgray', label='Data Points')
plt.plot(X, y_linear_pred, color='blue', label='Linear Regression')
plt.plot(X, y_nn_pred, color='red', label='Neural Network')
plt.title('Linear Regression vs. Neural Network Fit for $y = x^2$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True)
plt.show()