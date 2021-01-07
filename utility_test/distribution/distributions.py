from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class Gauss:
    def __init__(self, data):
        self.data = data
        mu, std = norm.fit(self.data)
        self.mu = mu
        self.std = std

    def distribution_parameter(self):
        return self.mu, self.std

    def save(self, output_path, title=None):
        xmin = min(self.data)
        xmax = max(self.data)
        x = np.linspace(xmin, xmax)

        p = norm.pdf(x, self.mu, self.std)
        plt.plot(x, p, 'k', linewidth=2, color='b')
        if title is not None:
            title = title + f" mu = {self.mu:.2f},  std = {self.std:.2f}"
        else:
            title = f"Gauss distr. mu = {self.mu:.2f},  std = {self.std:.2f}"

        plt.title(title)
        plt.savefig(output_path)
        plt.close()
