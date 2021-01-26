import tensorflow as tf

class Model:
    def __init__(self):
        self.targets = []
        self.outputs = []

    def process_example(self, target, data):
        self.targets.append(target)
        self.outputs.append(self.predict(data))

    def calculate_mse(self):
        self.mse = tf.keras.losses.MeanSquaredError()
        return self.mse(self.targets, self.outputs).numpy()

    def predict(self, x):
        raise NotImplementedError('Use a subclass instance implementing this method')


class BadExampleException(Exception):
    def __init__(self):
        message = 'Bad structure for given example: it must contain key and data each composed of 2 word'
        super().__init__(message)