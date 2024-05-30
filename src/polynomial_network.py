import math

import numpy as np
import matplotlib.pyplot as plt


class PolynomialNetwork:
    def __init__(self, power, eta=0.00035, epoch=15000, batch_size=50, lambda_=0.01):
        self.power = power  # Degree of the polynomial
        self.weights = np.random.rand(power + 1)  # Weights for the polynomial
        self.eta = eta  # Learning rate
        self.epoch = epoch  # Number of iterations
        self.batch_size = batch_size  # Batch size for gradient descent
        self.lambda_ = lambda_  # Regularization parameter

    def predict(self, x):
        prediction = sum([w * x ** i for i, w in enumerate(self.weights)])
        if math.isnan(prediction):
            raise ValueError("1 NaN values encountered during prediction. Try reducing the learning parameters.")
        return prediction

    def derivatives(self, x, y):
        c = 2 * (self.predict(x) - y)
        derivatives = [c * x ** i + self.lambda_ * w for i, w in enumerate(self.weights)]
        if np.isnan(derivatives).any():
            raise ValueError("2 NaN values encountered during derivatives calculation. Try reducing the learning "
                             "parameters.")
        return derivatives

    def train(self, x, y):
        # losses = []  # List to store loss values

        try:
            for e in range(self.epoch):
                # Combine x and y into a single numpy array
                data = np.array(list(zip(x, y)))
                # Randomly shuffle the data
                np.random.shuffle(data)

                for i in range(0, len(data), self.batch_size):
                    derivatives = np.array([0] * (self.power + 1), dtype=float)

                    for j in range(i, min(i + self.batch_size, len(data))):
                        # Split the shuffled data back into x and y
                        x_j, y_j = data[j]
                        derivatives += np.array(self.derivatives(x_j, y_j))

                    # derivatives = np.clip(derivatives, -1.8e308, 1.8e308)  # Clip the values to prevent overflow
                    self.weights -= self.eta / self.batch_size * derivatives

                loss = self.evaluate(x, y)
                # losses.append(loss)  # for plotting
                print(f"Progress:{round(e / self.epoch * 100, 2)}%, Loss:{round(loss, 6)}")
        except Exception as e:
            raise e

        # self.plot_loss(losses)

    def evaluate(self, x, y):
        return sum([abs(self.predict(x[i]) - y[i]) for i in range(len(x))]) / len(x)

    # plot the loss function
    @staticmethod
    def plot_loss(loss):
        fig, ax = plt.subplots()
        ax.plot(range(len(loss)), loss, 'g-')
        ax.set_xlabel('Batch number')
        ax.set_ylabel('Error')
        ax.set_title('Errors over time')
        plt.show()

    def get_weights(self):
        return self.weights.copy()
