import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import csv
import os

from similarity_pedersen.collect_pedersen_similarities import *
from base_model.BaselineAdditiveModel import BaselineAdditiveModel
from preprocessing.w2v_preprocessing_embedding import PreprocessingWord2VecEmbedding, OOVWordException
from utility_test.oracle.oracle import POSAwareOracle
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from utility_test.tester.tester import Tester, TestWriter, LineReader, UnexpectedValueInLine
from writer_reader_of_examples.writer_utility import Parser
from preprocessing.w2v_preprocessing_embedding import POS

"""
class CS10LineReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[4])
            first = line[0:2]
            second = line[2:4]

            return value, first, second
        except ValueError:
            raise UnexpectedValueInLine(line)
"""


class PedersenLineReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[11])
            first = line[2:4]
            second = line[12]
            target_pos = line[6]
            w1_pos = line[7]
            w2_pos = line[8]

            return value, first, second, target_pos, w1_pos, w2_pos
        except ValueError:
            raise UnexpectedValueInLine(line)


class CorrelationCouplesOracle(POSAwareOracle):
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
                    value, first, second, target_pos, w1_pos, w2_pos = reader.readline(line)
                    super().add_correlations(value, first, second, target_pos, w1_pos, w2_pos)
                except UnexpectedValueInLine:
                    continue


"""
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
"""


class PetersenEmbeddedOracle:
    def __init__(self, oracle: POSAwareOracle, preprocessor: PreprocessingWord2VecEmbedding):
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


"""
class CS10Tester(Tester):
    def __init__(self, embedded_oracle: CS10EmbeddedOracle):
        self.embedded_oracle = embedded_oracle

    def test_similarity_of_predictions(self, model, evaluator: SimilarityEvaluator, save_on_file=True):
        similarities = {}
        header = '\t'.join(['id', 'first_couple', 'second_couple', 'oracle_value', 'model_value', '#\n'])
        if save_on_file:
            writer_reader_of_examples = TestWriter('data/correlations_' + str(type(model).__name__) + '.txt', header)

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
                writer_reader_of_examples.write_line(i, [line['first'], line['second']], [str(line['value']), str(similarities[i])])

        if save_on_file:
            writer_reader_of_examples.release()

        return similarities

    def spearman_correlation_model_predictions_and_oracle(self, model, evaluator: SimilarityEvaluator,
                                                          save_on_file=True):
        similarities = self.test_similarity_of_predictions(model, evaluator, save_on_file)

        return spearmanr([similarities[x] for x in similarities], [self.embedded_oracle.oracle.correlations[x]['value']
                                                                   for x in self.embedded_oracle.oracle.correlations])
"""


class PetersenOOVTester(Tester):
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
                correlation = self.embedded_oracle.oracle.correlations[i]
                prediction_1 = model.predict(
                    [np.array([correlation['first_embedded'][0]]),
                     np.array([POS.get_pos_vector(correlation['w1_pos'])]),
                     np.array([correlation['first_embedded'][1]]),
                     np.array([POS.get_pos_vector(correlation['w2_pos'])]),
                     np.array([POS.get_pos_vector(correlation['target_pos'])])
                     ])
            prediction_2 = self.embedded_oracle.oracle.correlations[i]['second_embedded']
            similarities[i] = evaluator.similarity_function(prediction_1, prediction_2)[0]

            if save_on_file:
                line = self.embedded_oracle.oracle.correlations[i]
                writer.write_line(i, [line['first'], line['second']], [str(line['value']), str(similarities[i])])

        if save_on_file:
            writer.release()

        return similarities

    def spearman_correlation_model_predictions_and_oracle(self, model, evaluator: SimilarityEvaluator,
                                                          save_on_file=True):
        similarities = self.test_similarity_of_predictions(model, evaluator, save_on_file)
        """l1 = [x for x in similarities]
        l2 = [x for x in self.embedded_oracle.oracle.correlations]

        equals = True
        for i in range(0, len(l1)):
            if l1[i] != l2[i]:
                equals = False
                break
        print(f'THEY ARE EQUALS === {equals}')"""
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


def merge(n, definitions_path, oov_path, output_path):
    with open(definitions_path, 'r') as f1:
        with open(oov_path, 'r') as f2:
            with open(output_path, 'w+') as output:
                definitions = [definition.split('\t') for definition in f1.readlines()]
                oovs = [oov.split('\t') for oov in f2.readlines()]

                definitions.sort(key=lambda x: x[5])
                oovs.sort(key=lambda x: x[0])

                for definition in definitions:
                    definition.pop()
                for oov in oovs:
                    oov.pop()

                i = 1
                j = 1
                while i < len(definitions):
                    if random.uniform(0, 1) < n / len(definitions):
                        while j < len(oovs) and definitions[i][5] != oovs[j][0]:
                            j += 1

                        if j < len(oovs):
                            newline = '\t'.join(definitions[i] + oovs[j] + ['#\n'])
                            output.write(newline)
                            j += 1
                        else:
                            j = 1
                    i += 1


def write_test_targets(n, positives_input_path, negatives_input_path, output_path, measure):
    positive_def = 'data/test_similarity_pedersen/' + measure + '_positives_def.txt'
    negative_def = 'data/test_similarity_pedersen/' + measure + '_negatives_def.txt'
    merge(int(n / 2), 'data/similarity_pedersen_test/oov_pedersen_definition_pos_aware.txt', positives_input_path, positive_def)
    merge(int(n / 2), 'data/similarity_pedersen_test/oov_pedersen_definition_pos_aware.txt', negatives_input_path, negative_def)
    append(positive_def, negative_def, output_path)

    os.remove(positive_def)
    os.remove(negative_def)


"""
oracle = CorrelationCouplesOracle('data/CS10_test/AN_VO_CS10_test.txt')
oracle.collect_correlations(CS10LineReader(), range(1, 6))

embedded_oracle = CS10EmbeddedOracle(oracle, PreprocessingWord2VecEmbedding(
    "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))
"""


def test():
    # sequential: tf.keras.model_mappings.Sequential = tf.keras.model_mappings.load_model('oov_sequential_predictor.h5')
    evaluator = SimilarityEvaluator('cosine_similarity')
    functional = tf.keras.models.load_model('oov_functional_predictor.h5')
    baseline: BaselineAdditiveModel = BaselineAdditiveModel()

    tests_functional = []
    tests_additive = []

    for i in range(0, 10):
        output_path = 'data/similarity_pedersen_test/wup_oov_def.txt'
        write_test_targets(positives_input_path='data/similarity_pedersen_test/positive_wup_oov.txt',
                           negatives_input_path='data/similarity_pedersen_test/negative_wup_oov.txt', output_path=output_path)

        oracle = CorrelationCouplesOracle(output_path)
        oracle.collect_correlations(PedersenLineReader(), range(0, 18))

        embedded_oracle = PetersenEmbeddedOracle(oracle, preprocessor=PreprocessingWord2VecEmbedding(
            "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))

        tester = PetersenOOVTester(embedded_oracle)

        # spearman_sequential = tester.spearman_correlation_model_predictions_and_oracle(sequential, evaluator)
        spearman_functional = tester.spearman_correlation_model_predictions_and_oracle(functional, evaluator,
                                                                                       save_on_file=False)
        spearman_additive = tester.spearman_correlation_model_predictions_and_oracle(baseline, evaluator,
                                                                                     save_on_file=False)

        # print(str(type(sequential).__name__) + ' --> ' + str(spearman_sequential))
        print('--------------')
        print(str(type(functional).__name__) + ' --> ' + str(spearman_functional))
        print(str(type(baseline).__name__) + ' --> ' + str(spearman_additive))
        print('--------------')

        tests_functional.append(- spearman_functional.correlation)
        tests_additive.append(- spearman_additive.correlation)

    print(tests_functional)
    print(tests_additive)

    with open('data/similarity_pedersen_test/results_spearman_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'functional', 'additive'])
        writer.writerows([[i, tests_functional[i], tests_additive[i]] for i in range(0, len(tests_additive))])

    ax = plt.gca()
    ax.scatter([i for i in range(0, len(tests_functional))], tests_functional, color="b")
    ax.scatter([i for i in range(0, len(tests_additive))], tests_additive, color="r")
    plt.title('spearman values')
    plt.ylabel('spearman corr. coeff.')
    plt.xlabel('test #')
    plt.show()


def tests():
    evaluator = SimilarityEvaluator('cosine_similarity')
    functional = tf.keras.models.load_model('oov_functional_predictor.h5')
    baseline: BaselineAdditiveModel = BaselineAdditiveModel()

    tests_baseline_by_measures = {}
    tests_functional_by_measures = {}

    similarities_function_names = ['path', 'wup', 'lch', 'res', 'jcn', 'lin']

    preprocessor = PreprocessingWord2VecEmbedding(
        'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    for measure in similarities_function_names:
        tests_functional_by_measures[measure] = []
        tests_baseline_by_measures[measure] = []

        positive_similarities = 'data/similarity_pedersen_test/oov_similarities/' + measure + '_positive_couples.txt'
        negative_similarities = 'data/similarity_pedersen_test/oov_similarities/' + measure + '_negative_couples.txt'
        oov_sim(measure,
                positive_output_file=positive_similarities,
                negative_output_file=negative_similarities)
        for i in range(0, 10):
            output_path = 'data/similarity_pedersen_test/oov_similarities/' + measure + '_' + str(i) + '_oov_def.txt'
            write_test_targets(n=1000, positives_input_path=positive_similarities,
                               negatives_input_path=negative_similarities, output_path=output_path, measure=measure)

            oracle = CorrelationCouplesOracle(output_path)
            oracle.collect_correlations(PedersenLineReader(), range(0, 16))

            embedded_oracle = PetersenEmbeddedOracle(oracle, preprocessor=preprocessor)
            tester = PetersenOOVTester(embedded_oracle)

            # spearman_sequential = tester.spearman_correlation_model_predictions_and_oracle(sequential, evaluator)
            spearman_functional = tester.spearman_correlation_model_predictions_and_oracle(functional, evaluator,
                                                                                           save_on_file=False)
            spearman_additive = tester.spearman_correlation_model_predictions_and_oracle(baseline, evaluator,
                                                                                         save_on_file=False)

            # print(str(type(sequential).__name__) + ' --> ' + str(spearman_sequential))
            print('--------------')
            print(str(type(functional).__name__) + ' --> ' + str(spearman_functional))
            print(str(type(baseline).__name__) + ' --> ' + str(spearman_additive))
            print('--------------')

            print(measure, - spearman_functional.correlation)
            print(measure, - spearman_additive.correlation)

            tests_functional_by_measures[measure].append(- spearman_functional.correlation)
            tests_baseline_by_measures[measure].append(- spearman_additive.correlation)

    with open('data/similarity_pedersen_test/oov_similarities/results_spearman_test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['measure', 'x', 'functional', 'additive'])
        writer.writerows([[measure, i, tests_functional_by_measures[measure][i], tests_baseline_by_measures[measure][i]]
                          for measure in tests_functional_by_measures for i in
                          range(0, len(tests_functional_by_measures[measure]))])
