import tensorflow as tf
import numpy as np
import random
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


class PedersenLineReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[11])
            first = line[2:4]
            second = line[10]
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


class CS10EmbeddedOracle:
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


class PetersenEmbeddedOracle:
    def __init__(self, oracle: Oracle, preprocessor: PreprocessingWord2VecEmbedding):
        self.oracle = oracle

        delete = []
        for id in oracle.correlations:
            try:
                self.oracle.correlations[id]['first_embedded'] = np.array([preprocessor.get_vector(word=word)
                                                                           for word in
                                                                           oracle.correlations[id]['first']])
                self.oracle.correlations[id]['second_embedded'] = preprocessor.get_vector(
                    word=oracle.correlations[id]['second'])
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
    def test_similatity_of_predictions(self, model, evaluator: SimilarityEvaluator, save_on_file=True):
        pass

    def spearman_correlation_model_predictions_and_oracle(self, model, evaluator: SimilarityEvaluator,
                                                          save_on_file=True):
        pass


class CS10Tester(Tester):
    def __init__(self, embedded_oracle: CS10EmbeddedOracle):
        self.embedded_oracle = embedded_oracle

    def test_similarity_of_predictions(self, model, evaluator: SimilarityEvaluator, save_on_file=True):
        similarities = {}
        header = '\t'.join(['id', 'first_couple', 'second_couple', 'oracle_value', 'model_value', '#\n'])
        if save_on_file:
            writer = TestWriter('data/correlations_' + str(type(model).__name__) + '.txt', header)

        for i in self.embedded_oracle.oracle.correlations:
            if type(model) is BaselineAdditiveModel or type(model) is tf.keras.Sequential:
                prediction_1 = model.predict(np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded']]))
                prediction_2 = model.predict(np.array([self.embedded_oracle.oracle.correlations[i]['second_embedded']]))
            else:
                prediction_1 = model.predict(
                    [np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded'][0]]),
                     np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded'][1]])
                     ])
                prediction_2 = model.predict(
                    [np.array([self.embedded_oracle.oracle.correlations[i]['second_embedded'][0]]),
                     np.array([self.embedded_oracle.oracle.correlations[i]['second_embedded'][1]])
                     ])

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


class PetersenTester(Tester):
    def __init__(self, embedded_oracle: PetersenEmbeddedOracle):
        self.embedded_oracle = embedded_oracle

    def test_similarity_of_predictions(self, model, evaluator: SimilarityEvaluator, save_on_file=True):
        similarities = {}
        header = '\t'.join(['id', 'first_couple', 'first', 'second', 'oracle_value', 'model_value', '#\n'])
        if save_on_file:
            writer = TestWriter('data/petersen_correlations_' + str(type(model).__name__) + '.txt', header)
        for i in self.embedded_oracle.oracle.correlations:
            if type(model) is BaselineAdditiveModel or type(model) is tf.keras.Sequential:
                prediction_1 = model.predict(np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded']]))
            else:
                prediction_1 = model.predict(
                    [np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded'][0]]),
                     np.array([self.embedded_oracle.oracle.correlations[i]['first_embedded'][1]])
                     ])
            prediction_2 = self.embedded_oracle.oracle.correlations[i]['second_embedded']
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


def append(file1, file2, ouput):
    with open(file1, 'r') as f1:
        with open(file2, 'r') as f2:
            out = open(ouput, 'w')
            l1 = f1.readlines()
            l2 = f2.readlines()

            out.writelines(l1 + l2)
            out.close()


def merge(definitions_path, oov_path, output_path):
    with open(definitions_path, 'r') as f1:
        with open(oov_path, 'r') as f2:
            output = open(output_path, 'w')
            definitions = f1.readlines()
            oovs = f2.readlines()

            i = 1
            while i < len(definitions):
                j = 1
                while j < len(oovs) and definitions[i].split('\t')[5] != oovs[j].split('\t')[0]:
                    j += 1
                if j < len(oovs) and random.uniform(0, 1) > 0.9:
                    output.write("{}\t{}\n".format(definitions[i].rstrip(), oovs[j].rstrip()))
                i += 1
            output.close()


def write_test_targets(positives_input_path, negatives_input_path, output_path):
    positive_def = 'data/pedersen_test/positives_def.txt'
    negative_def = 'data/pedersen_test/negatives_def.txt'
    merge('data/pedersen_test/oov_pedersen_definition.txt', positives_input_path, positive_def)
    merge('data/pedersen_test/oov_pedersen_definition.txt', negatives_input_path, negative_def)
    append(positive_def, negative_def, output_path)


"""
oracle = CorrelationCouplesOracle('data/CS10_test/AN_VO_CS10_test.txt')
oracle.collect_correlations(CS10LineReader(), range(1, 6))

embedded_oracle = CS10EmbeddedOracle(oracle, PreprocessingWord2VecEmbedding(
    "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))
"""

output_path = 'data/pedersen_test/wup_oov_def.txt'
write_test_targets(positives_input_path='data/pedersen_test/positive_wup_oov.txt',
                   negatives_input_path='data/pedersen_test/negative_wup_oov.txt', output_path=output_path)

oracle = CorrelationCouplesOracle(output_path)
oracle.collect_correlations(PedersenLineReader(), range(0, 13))

embedded_oracle = PetersenEmbeddedOracle(oracle, preprocessor=PreprocessingWord2VecEmbedding(
    "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))

tester = PetersenTester(embedded_oracle)
evaluator = SimilarityEvaluator('cosine_similarity')

sequential: tf.keras.models.Sequential = tf.keras.models.load_model('oov_sequential_predictor.h5')
functional = tf.keras.models.load_model('oov_functional_predictor.h5')
baseline: BaselineAdditiveModel = BaselineAdditiveModel()

spearman_sequential = tester.spearman_correlation_model_predictions_and_oracle(sequential, evaluator)
spearman_functional = tester.spearman_correlation_model_predictions_and_oracle(functional, evaluator)
spearman_additive = tester.spearman_correlation_model_predictions_and_oracle(baseline, evaluator)

print(str(type(sequential).__name__) + ' --> ' + str(spearman_sequential))
print(str(type(functional).__name__) + ' --> ' + str(spearman_functional))
print(str(type(baseline).__name__) + ' --> ' + str(spearman_additive))
