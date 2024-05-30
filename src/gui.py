import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import polynomial_network as pn


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plot Analyzer")

        style = ttk.Style()
        style.theme_use('clam')

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-50, 50)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        self.line, = self.ax.plot([], [], 'b-')
        self.xdata = []
        self.ydata = []

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(padx=10, pady=10)

        self.power_label = tk.Label(self.button_frame, text="Power")
        self.power_label.grid(row=0, column=0, padx=5, pady=5)
        self.power_entry = ttk.Combobox(self.button_frame, font=('Helvetica', 10))
        self.power_entry['values'] = [str(i) for i in range(1, 20)]
        self.power_entry.current(5)
        self.power_entry.grid(row=0, column=1, padx=5, pady=5)

        self.eta_label = tk.Label(self.button_frame, text="Eta")
        self.eta_label.grid(row=1, column=0, padx=5, pady=5)
        self.eta_entry = ttk.Entry(self.button_frame, font=('Helvetica', 10))
        self.eta_entry.insert(0, '0.00035')
        self.eta_entry.grid(row=1, column=1, padx=5, pady=5)

        self.epoch_label = tk.Label(self.button_frame, text="Epoch")
        self.epoch_label.grid(row=2, column=0, padx=5, pady=5)
        self.epoch_entry = ttk.Entry(self.button_frame, font=('Helvetica', 10))
        self.epoch_entry.insert(0, '15000')
        self.epoch_entry.grid(row=2, column=1, padx=5, pady=5)

        self.batch_size_label = tk.Label(self.button_frame, text="Batch Size")
        self.batch_size_label.grid(row=3, column=0, padx=5, pady=5)
        self.batch_size_entry = ttk.Entry(self.button_frame, font=('Helvetica', 10))
        self.batch_size_entry.insert(0, '50')
        self.batch_size_entry.grid(row=3, column=1, padx=5, pady=5)

        self.lambda_label = tk.Label(self.button_frame, text="Lambda")
        self.lambda_label.grid(row=4, column=0, padx=5, pady=5)
        self.lambda_entry = ttk.Entry(self.button_frame, font=('Helvetica', 10))
        self.lambda_entry.insert(0, '0.01')
        self.lambda_entry.grid(row=4, column=1, padx=5, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_button.grid(row=5, column=0, padx=5, pady=5)

        self.analyze_button = tk.Button(self.button_frame, text="Analyze", command=self.analyze)
        self.analyze_button.grid(row=5, column=1, padx=5, pady=5)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(padx=10, pady=10)

    def on_click(self, event):
        if event.inaxes != self.ax or (self.xdata and event.xdata < max(self.xdata)):
            return
        self.xdata.append(event.xdata)
        self.ydata.append(event.ydata)
        self.update_limes()
        self.update_plot()

    def on_motion(self, event):
        if event.inaxes != self.ax or not event.button or (self.xdata and event.xdata < max(self.xdata)):
            return
        self.xdata.append(event.xdata)
        self.ydata.append(event.ydata)
        self.update_limes()
        self.update_plot()

    def update_limes(self):
        if max(self.xdata) >= self.ax.get_xlim()[1] * 0.8:
            # Increase the x limit by 0.7% of the current limit
            self.ax.set_xlim(self.ax.get_xlim()[0], self.ax.get_xlim()[1] * 1.007)

        if max(self.ydata) >= self.ax.get_ylim()[1] * 0.8:
            # Increase the y limit by 0.7% of the current limit
            self.ax.set_ylim(self.ax.get_ylim()[0], self.ax.get_ylim()[1] * 1.007)

        if min(self.ydata) <= self.ax.get_ylim()[0] * 0.8:
            # Decrease the y limit by 0.7% of the current limit
            self.ax.set_ylim(self.ax.get_ylim()[0] * 1.007, self.ax.get_ylim()[1])

    def update_plot(self):
        self.line.set_data(self.xdata, self.ydata)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def clear(self):
        self.xdata = []
        self.ydata = []
        self.ax.clear()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-50, 50)
        self.line, = self.ax.plot([], [], 'b-')
        self.update_plot()

    def analyze(self):
        if len(self.xdata) == 0 or len(self.ydata) == 0:
            self.result_label.config(text="No data to analyze")
            return

        self.result_label.config(text="Analyzing data...")

        xdata = np.array(self.xdata)
        ydata = np.array(self.ydata)

        # Normalize data
        x_mean, x_std = np.mean(xdata), np.std(xdata)
        y_mean, y_std = np.mean(ydata), np.std(ydata)
        xdata_norm = (xdata - x_mean) / x_std
        ydata_norm = (ydata - y_mean) / y_std

        try:
            power = int(self.power_entry.get())
            eta = float(self.eta_entry.get())
            epoch = int(self.epoch_entry.get())
            batch_size = int(self.batch_size_entry.get())
            lambda_ = float(self.lambda_entry.get())
        except ValueError:
            self.result_label.config(text="Please enter valid numeric values")
            return

        network = pn.PolynomialNetwork(power=power, eta=eta, epoch=epoch, batch_size=batch_size, lambda_=lambda_)

        try:
            network.train(xdata_norm, ydata_norm)
        except Exception as e:
            self.result_label.config(text=str(e))
            return

        # Generate dense x-values for plotting
        # dense_x = np.linspace(min(xdata_norm), max(xdata_norm), 300)
        # predicted_y_norm = [network.predict(x) for x in dense_x]

        # predicted_y_norm = np.array(predicted_y_norm)

        # Denormalize data for plotting
        # dense_x_orig = dense_x * x_std + x_mean
        # predicted_y_orig = predicted_y_norm * y_std + y_mean

        # fig, ax = plt.subplots()
        # ax.scatter(xdata, ydata, label='Data')
        # ax.plot(dense_x_orig, predicted_y_orig, 'r-', label='Fitted polynomial')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_title('Polynomial Fit')
        # ax.legend()
        # plt.show()

        weights_norm = network.get_weights()
        # print(f"Learned Weights (normalized): {weights_norm}")

        # Transform weights back to original scale
        weights_orig = np.zeros_like(weights_norm)
        for i in range(len(weights_norm)):
            weights_orig[i] = weights_norm[i] * (y_std / (x_std ** i))

        # Adjust intercept
        weights_orig[0] += y_mean - sum([weights_orig[i] * (x_mean ** i) for i in range(1, len(weights_orig))])
        # print(f"Learned Weights (original): {weights_orig}")

        # Create a string that represents the polynomial equation
        polynomial_str = "y = "
        for i, w in enumerate(weights_orig):
            if i == 0:
                polynomial_str += f"{w:.4f}"
            else:
                polynomial_str += f" + {w:.4f}*x^{i}"

        self.result_label.config(text=f"Learned Weights (original): {weights_orig}\nPolynomial: {polynomial_str}")

        # Define polynomial function with transformed weights
        def polynomial(x, weights):
            return sum([w * x ** i for i, w in enumerate(weights)])

        # Generate y-values for the original xdata using the fitted polynomial
        y_fitted = [polynomial(x, weights_orig) for x in self.xdata]

        # Plot the original data and the fitted polynomial in different plot
        # nfig, nax = plt.subplots()
        # nax.plot(xdata, ydata, label='Original data')
        # nax.plot(xdata, y_fitted, 'r-', label='Fitted polynomial')
        # nax.set_xlabel('x')
        # nax.set_ylabel('y')
        # nax.set_title('Polynomial Fit')
        # nax.legend()
        # plt.show()

        # Clear the current plot
        self.ax.clear()

        # Draw the original data and the fitted polynomial
        self.ax.plot(xdata, ydata, label='Original data')
        self.ax.plot(xdata, y_fitted, 'r-', label='Fitted polynomial')

        # Set labels and title
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Polynomial Fit')

        # Show legend
        self.ax.legend()

        # Redraw the canvas
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
