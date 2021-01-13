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

# TODO You should define new class for CS10 test and delete the other classes: OOV tests are in similarity_pedersen dir

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

"""
oracle = CorrelationCouplesOracle('data/CS10_test/AN_VO_CS10_test.txt')
oracle.collect_correlations(CS10LineReader(), range(1, 6))

embedded_oracle = CS10EmbeddedOracle(oracle, PreprocessingWord2VecEmbedding(
    "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True))
"""
