import tensorflow as tf
from embedding_utilities import BaselineAdditiveModel, Parser, PreprocessingWord2VecEmbedding, OOVWordException
import numpy as np
from scipy.stats import spearmanr


class Oracle:
    def __init__(self):
        self.correlations = {}

    def add_correlations(self, value, first, second):
        self.correlations[len(self.correlations)] = {'value': value, 'first': first, 'second': second}


class CorrelationCouplesOracle(Oracle):
    def __init__(self, path, value_index=0, first_index=0, first_end=None, second_index=0, second_end=None):
        self.path = path
        super().__init__()
        self.collect_correlations(value_index, first_index, first_end, second_index, second_end)

    def collect_correlations(self, value_index=0, first_index=0, first_end=None, second_index=0, second_end=None):
        parser = Parser(self.path, '\t')
        index_range = range(1, 6)

        with parser:
            while True:
                line = parser.get_example_from_line_next_line(index_range)
                if not line:
                    break

                try:
                    value = float(line[value_index])
                    if first_end is not None:
                        first = line[first_index:first_end]
                    else:
                        first = [line[first_index]]

                    if second_end is not None:
                        second = line[second_index:second_end]
                    else:
                        second = [line[second_index]]

                    super().add_correlations(value, first, second)
                except ValueError:
                    continue


class EmbeddedOracle:
    def __init__(self, oracle: Oracle, preprocessor: PreprocessingWord2VecEmbedding):
        self.oracle = oracle

        delete = []
        for id in oracle.correlations:
            try:
                self.oracle.correlations[id]['first_embedded'] = [preprocessor.get_vector(word=word) for word in
                                                                  oracle.correlations[id]['first']]
                self.oracle.correlations[id]['second_embedded'] = [preprocessor.get_vector(word=word) for word in
                                                                   oracle.correlations[id]['second']]
            except OOVWordException:
                delete.append(id)

        for id in delete:
            if id in self.oracle.correlations:
                del self.oracle.correlations[id]


class SimilarityEvaluator:
    def __init__(self, similarity_function):
        if similarity_function == 'cosine_similarity':
            self.similarity_function = tf.keras.losses.cosine_similarity
        else:
            raise ValueError('Unknown similarity function')


class Tester:
    def __init__(self, embedded_oracle: EmbeddedOracle):
        self.embedded_oracle = embedded_oracle

    def test_similarity_of_predictions(self, model, evaluator: SimilarityEvaluator):

        first_couples_predictions = model.predict(
            np.array([np.array(self.embedded_oracle.oracle.correlations[x]['first_embedded'])
                      for x in self.embedded_oracle.oracle.correlations]))
        second_couple_predictions = model.predict(
            np.array([np.array(self.embedded_oracle.oracle.correlations[x]['second_embedded'])
                      for x in self.embedded_oracle.oracle.correlations]))

        similarities = []
        for i in range(0, len(first_couples_predictions)):
            similarities.append(
                evaluator.similarity_function(first_couples_predictions[i], second_couple_predictions[i]).numpy())
        return similarities

    def spearman_correlation_model_predictions_and_oracle(self, model, evaluator: SimilarityEvaluator):
        predicted_correlation = self.test_similarity_of_predictions(model, evaluator)

        return spearmanr(predicted_correlation, [self.embedded_oracle.oracle.correlations[x]['value']
                                                 for x in self.embedded_oracle.oracle.correlations])


oracle = CorrelationCouplesOracle('data/CS10_test/a/AN_CS10_test.txt', 4, 0, 2, 2, 4)
print(oracle.correlations)
embedded_oracle = EmbeddedOracle(oracle,
                                 PreprocessingWord2VecEmbedding(
                                     "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))

tester = Tester(embedded_oracle)
evaluator = SimilarityEvaluator('cosine_similarity')

model: tf.keras.models.Sequential = tf.keras.models.load_model('oov_sequential_predictor.h5')
baseline: BaselineAdditiveModel = BaselineAdditiveModel()

print(str(type(model)) + '-->')
print(tester.spearman_correlation_model_predictions_and_oracle(model, evaluator))
print(str(type(baseline)) + '-->')
print(tester.spearman_correlation_model_predictions_and_oracle(baseline, evaluator))
