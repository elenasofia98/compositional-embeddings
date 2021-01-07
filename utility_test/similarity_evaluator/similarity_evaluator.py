import tensorflow as tf
from enum import Enum


class SimilarityFuction(Enum):
    cosine_similarity = lambda x, y: tf.keras.losses.cosine_similarity(x, y).numpy()


class SimilarityEvaluator:
    def __init__(self, similarity_function):
        if similarity_function == 'cosine_similarity':
            self.similarity_function = SimilarityFuction.cosine_similarity
        else:
            raise ValueError('Unknown similarity function')