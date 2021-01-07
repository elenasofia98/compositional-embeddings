import numpy as np
from base_model.model import BadExampleException, Model


class BaselineAdditiveModel(Model):
    def __init__(self):
        super(BaselineAdditiveModel, self).__init__()

    def process_example(self, target, data):
        if len(data) != 2:
            raise BadExampleException
        super(BaselineAdditiveModel, self).process_example(target, data)

    def predict(self, x):
        if x.shape == (2, 300):
            return np.sum(a=x, axis=0)
        predictions = []
        for a in x:
            predictions.append(np.sum(a=a, axis=0))
        return np.array(predictions)