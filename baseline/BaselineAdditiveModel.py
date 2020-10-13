import numpy as np
import tensorflow as tf
from example_to_numpy.example_to_numpy import BadExampleException

class BaselineAdditiveModel:
    def __init__(self):
        self.targets = []
        self.outputs = []

    def process_example(self, target, data):
        if len(data) != 2:
            raise BadExampleException

        self.targets.append(target)
        self.outputs.append(self.predict(data))

    def calculate_mse(self):
        self.mse = tf.keras.losses.MeanSquaredError()
        return self.mse(self.targets, self.outputs).numpy()

    def predict(self, x):
        if x.shape == (2, 300):
            return np.sum(a=x, axis=0)
        predictions = []
        for a in x:
            predictions.append(np.sum(a=a, axis=0))
        return np.array(predictions)