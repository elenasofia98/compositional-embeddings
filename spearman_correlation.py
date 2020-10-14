import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr

from baseline.BaselineAdditiveModel import BaselineAdditiveModel
from preprocessing.w2v_preprocessing_embedding import PreprocessingWord2VecEmbedding, OOVWordException
from writer.writer_utility import Parser


class Oracle:
    def __init__(self):
        self.correlations = {}

    def add_correlations(self, value, first, second):
        self.correlations[len(self.correlations)] = {'value': value, 'first': first, 'second': second}


class UnexpectedValueInLine(ValueError):
    def __init__(self, line):
        message = 'Value Error occurred during the reading of line:\n' + str(line)
        super().__init__(message)


class LineReader:
    def readline(self, line):
        pass


class CS10LineReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[4])
            first = line[0:2]
            second = line[2:4]

            return value, first, second
        except ValueError:
            raise UnexpectedValueInLine(line)


class CorrelationCouplesOracle(Oracle):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def collect_correlations(self, reader: LineReader, index_range: range):
        parser = Parser(self.path, '\t')
        with parser:
            while True:
                line = parser.get_example_from_line_next_line(index_range)
                if not line:
                    break

                try:
                    value, first, second = reader.readline(line)
                    super().add_correlations(value, first, second)
                except UnexpectedValueInLine:
                    continue


class EmbeddedOracle:
    def __init__(self, oracle: Oracle, preprocessor: PreprocessingWord2VecEmbedding):
        self.oracle = oracle

        delete = []
        for id in oracle.correlations:
            try:
                self.oracle.correlations[id]['first_embedded'] = np.array([preprocessor.get_vector(word=word)
                                                                           for word in
                                                                           oracle.correlations[id]['first']])
                self.oracle.correlations[id]['second_embedded'] = np.array([preprocessor.get_vector(word=word)
                                                                            for word in
                                                                            oracle.correlations[id]['second']])
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


class TestWriter:
    def __init__(self, path, header):
        self.create_file(path)
        self.write_header(header)

        self.separator = '\t'

    def write_header(self, sequence):
        self.file.write(sequence)

    def create_file(self, path):
        self.file = open(path, 'w+')

    def write_line(self, index, oracle_line, correlations):
        oracle_line = self.separator.join([str(x) for x in oracle_line])
        correlations = self.separator.join([str(x) for x in correlations])

        self.file.write(self.separator.join([str(index), oracle_line, correlations, '#\n']))

    def write_lines(self, lines):
        self.file.writelines(lines)

    def release(self):
        self.file.close()


class Tester:
    def __init__(self, embedded_oracle: EmbeddedOracle):
        self.embedded_oracle = embedded_oracle

    def test_similarity_of_predictions(self, model, evaluator: SimilarityEvaluator, save_on_file=True):
        similarities = {}
        header = '\t'.join(['id', 'first_couple', 'second_couple', 'oracle_value', 'model_value', '#\n'])
        if save_on_file:
            writer = TestWriter('data/correlations_' + str(type(model).__name__) + '.txt', header)

        for i in self.embedded_oracle.oracle.correlations:
            prediction_1 = model.predict(np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded']]))
            prediction_2 = model.predict(np.array([self.embedded_oracle.oracle.correlations[i]['second_embedded']]))
            similarities[i] = evaluator.similarity_function(prediction_1, prediction_2).numpy()[0]

            if save_on_file:
                line = self.embedded_oracle.oracle.correlations[i]
                writer.write_line(i, [line['first'], line['second']], [str(line['value']), str(similarities[i])])

        if save_on_file:
            writer.release()

        return similarities

    def spearman_correlation_model_predictions_and_oracle(self, model, evaluator: SimilarityEvaluator,
                                                          save_on_file=True):
        similarities = self.test_similarity_of_predictions(model, evaluator, save_on_file)

        return spearmanr([similarities[x] for x in similarities], [self.embedded_oracle.oracle.correlations[x]['value']
                                                                   for x in self.embedded_oracle.oracle.correlations])


oracle = CorrelationCouplesOracle('data/CS10_test/ALL_CS10_test.txt')
oracle.collect_correlations(CS10LineReader(), range(1, 6))

embedded_oracle = EmbeddedOracle(oracle, PreprocessingWord2VecEmbedding(
    "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))

tester = Tester(embedded_oracle)
evaluator = SimilarityEvaluator('cosine_similarity')

model: tf.keras.models.Sequential = tf.keras.models.load_model('oov_sequential_predictor.h5')
baseline: BaselineAdditiveModel = BaselineAdditiveModel()

spearman_model = tester.spearman_correlation_model_predictions_and_oracle(model, evaluator)
spearman_additive = tester.spearman_correlation_model_predictions_and_oracle(baseline, evaluator)

print(str(type(model).__name__) + ' --> ' + str(spearman_model))
print(str(type(baseline).__name__) + ' --> ' + str(spearman_additive))
