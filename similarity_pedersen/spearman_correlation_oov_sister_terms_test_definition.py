import random

import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
import numpy as np
from gensim.models import KeyedVectors, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors
from scipy.stats import spearmanr
import os

from base_model.ParentModel import ParentModel

from base_model.BaselineAdditiveModel import BaselineAdditiveModel
from cluster.cluster import ClusterMinDiam
from similarity_pedersen.collect_pedersen_similarities import retrieve_oov_couples_divided_by_value_of_similarity
from similarity_pedersen.pedersen_similarities import SimilarityFunction, Comparator, ReaderSynsetCouples, \
    SynsetOOVCouple
from utility_test.distribution.distributions import Gauss
from utility_test.oracle.oracle import POSAwareOracle, Oracle, POSAwareOOVOracle
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from utility_test.tester.tester import Tester, TestWriter, LineReader, UnexpectedValueInLine
from word_in_vocabulary import WordInSynset, Checker
from writer_reader_of_examples.writer_utility import Parser
from preprocessing.w2v_preprocessing_embedding import POS


class DefinitionsOOVSisterTerms_Joiner:
    def __init__(self, definitions_path, sister_terms_path, output_path):
        self.definitions_path = definitions_path
        self.sister_terms_path = sister_terms_path
        self.output_path = output_path

    def join(self):
        with open(self.definitions_path, 'r') as definitions_file:
            with open(self.sister_terms_path, 'r') as sister_terms_file:
                sister_terms = []
                first = True
                for line in sister_terms_file.readlines():
                    if first:
                        first = False
                        continue

                    split = line.split('\t')
                    oov_s, w2_s2 = WordInSynset(word=split[2], synset_name=split[0], pos=split[4]), WordInSynset(
                        word=split[3], synset_name=split[1], pos=split[4])
                    sister_terms.append((oov_s, w2_s2))

                definitions = []
                first = True
                for line in definitions_file.readlines():
                    if first:
                        first = False
                        continue

                    split = line.split('\t')
                    oov_s = WordInSynset(word=split[1], synset_name=split[5], pos=split[6])
                    split.pop()
                    definitions.append((oov_s, split))

                with open(self.output_path, 'w+') as output_file:
                    output_lines = []

                    for (oov_s, w2_s2) in sister_terms:
                        oov_s: WordInSynset = oov_s
                        for (candidate_oov_s, definition_line) in definitions:
                            if oov_s.equals(candidate_oov_s):
                                definition_line.extend([w2_s2.synset_name, w2_s2.word, '#\n'])
                                output_lines.append('\t'.join(definition_line))

                    output_file.writelines(output_lines)


class OOVSisterTermsSimilarity:
    def __init__(self, positive_input_path, negative_input_path, normalize_len=False):
        self.positive_input_path = positive_input_path
        self.negative_input_path = negative_input_path
        self.normalize_len = normalize_len

    def compare_according_to(self, similarity_function_name, root_output_dir):
        similarity_function = SimilarityFunction.by_name(similarity_function_name)
        file_name = 'oov_oracle_' + similarity_function_name + '.txt'

        comparator = Comparator(self._get_synset_couples(), similarity_function)
        similarities = comparator.get_similarities()

        if not os.path.exists(root_output_dir):
            os.mkdir(root_output_dir)

        with open(self.positive_input_path, 'r') as positive_file:
            with open(self.negative_input_path, 'r') as negative_file:
                output_lines = []
                with open(os.path.join(root_output_dir, file_name), 'w+') as output_file:
                    for line in positive_file.readlines():
                        split = line.split('\t')
                        split.pop()

                        s1_index = 5
                        w1_index = 1
                        s2_index = 9
                        w2_index = 10
                        s_pos_index = 6

                        # TODO you can omit controls like these cause the lists should be ordered in the same way, check if there are issue why this could be not true (errors, exceptions ecc)
                        for similarity in similarities:
                            if similarity[0] == split[s1_index] and similarity[1] == split[s2_index]:
                                output_lines.append('\t'.join(split + [similarity[4], '#\n']))
                                break

                    for line in negative_file.readlines():
                        split = line.split('\t')
                        split.pop()

                        # TODO you can omit controls like these cause the lists should be ordered in the same way, check if there are issue why this could be not true (errors, exceptions ecc)
                        for similarity in similarities:
                            if similarity[0] == split[s1_index] and similarity[1] == split[s2_index]:
                                output_lines.append('\t'.join(split + [similarity[4], '#\n']))
                                break

                    output_file.writelines(output_lines)

    def _get_synset_couples(self):
        positive_couples = ReaderSynsetCouples.read(self.positive_input_path,
                                                    s1_index=5, w1_index=1,
                                                    s2_index=9, w2_index=10, s_pos_index=6, exclude_first=False)
        negative_couples = ReaderSynsetCouples.read(self.negative_input_path,
                                                    s1_index=5, w1_index=1,
                                                    s2_index=9, w2_index=10, s_pos_index=6, exclude_first=False)

        if self.normalize_len:
            if len(positive_couples) <= len(negative_couples):
                r = len(positive_couples) / len(negative_couples)
                negative_couples = [x for x in negative_couples if random.uniform(0, 1) <= r]
            else:
                r = len(negative_couples) / len(positive_couples)
                positive_couples = [x for x in negative_couples if random.uniform(0, 1) <= r]
        return positive_couples + negative_couples


class OOVSisterTerms_LineReader(LineReader):
    def readline(self, line):
        s1_index = 5
        w1_index = 1
        s2_index = 9
        w2_index = 10
        s_pos_index = 6

        try:
            value = float(line[11])
            oov = line[w1_index]
            synset_oov = line[s1_index]
            first = line[2:4]
            second = line[w2_index]
            synset_second = line[s2_index]
            target_pos = line[s_pos_index]
            w1_pos = line[s_pos_index + 1]
            w2_pos = line[s_pos_index + 2]

            return value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos
        except ValueError:
            raise UnexpectedValueInLine(line)


class OOVSisterTerms_POSAwareOracle(POSAwareOOVOracle):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def add_correlations(self, value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos):
        self.correlations[len(self.correlations)] = {'value': value, 'oov': oov, 'synset_oov': synset_oov,
                                                     'first': first, 'second': second,
                                                     'synset_second': synset_second,
                                                     'target_pos': target_pos,
                                                     'w1_pos': w1_pos, 'w2_pos': w2_pos}

    def collect_correlations(self, reader: OOVSisterTerms_LineReader, index_range=range(0, 13)):
        parser = Parser(self.path, '\t')
        with parser:
            while True:
                line = parser.get_example_from_line_next_line(index_range)
                if not line:
                    break

                try:
                    value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos = reader.readline(
                        line)
                    self.add_correlations(value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos,
                                          w2_pos)
                except UnexpectedValueInLine:
                    continue

    def remove_correlations_with_oov(self, checker: Checker):
        del_keys = []
        for key in self.correlations:
            correlation = self.correlations[key]
            if not checker.is_in_vocabulary(correlation['first'][0]) or not checker.is_in_vocabulary(
                    correlation['first'][1]) or not checker.is_in_vocabulary(correlation['second']):
                del_keys.append(key)
        for key in del_keys:
            self.correlations.pop(key)


class OOVSisterTerms_POSAwareTester(Tester):
    def __init__(self, oracle: OOVSisterTerms_POSAwareOracle):
        self.oracle = oracle

    def collect_similarity_of_predictions(self, test_model, evaluator: SimilarityEvaluator,
                                          save_on_file, path, mode,
                                          pretrained_embeddings_model):
        similarities = {}

        if save_on_file:
            header = '\t'.join(
                ['id', 'oov', 'synset_oov', 'first', 'second', 'synset_second', 'target_pos', 'w1_pos', 'w2_pos',
                 'oracle_value', 'model_value' '#\n'])
            writer = TestWriter(path, header, mode)

        for i in self.oracle.correlations:
            correlation = self.oracle.correlations[i]

            prediction_1 = self._predict_according_to(test_model, pretrained_embeddings_model, correlation)
            prediction_2 = pretrained_embeddings_model.word_vec(correlation['second'])

            similarities[i] = evaluator.similarity_function(np.array(prediction_1), np.array(prediction_2))

            if save_on_file:
                writer.write_free_line(index=i, line=[correlation['oov'], correlation['synset_oov'],
                                                      '[' + correlation['first'][0] + ' ' + correlation['first'][
                                                          1] + ']',
                                                      correlation['second'], correlation['synset_second'],
                                                      correlation['target_pos'], correlation['w1_pos'],
                                                      correlation['w2_pos'],
                                                      str(correlation['value']), str(similarities[i])])

        if save_on_file:
            writer.release()

        return similarities

    def _predict_according_to(self, test_model, pretrained_embeddings_model, correlation):
        if isinstance(test_model, ParentModel):
            pred = test_model.predict(correlation['oov'], pos_tag=correlation['target_pos'].lower())
            return pred

        if isinstance(test_model, KeyedVectors) or isinstance(test_model, FastTextKeyedVectors):
            pred = test_model.word_vec(correlation['oov'])
            return pred

        first_embeddings = np.array([pretrained_embeddings_model.word_vec(word) for word in correlation['first']])
        if isinstance(test_model, BaselineAdditiveModel):
            pred = test_model.predict(first_embeddings)
            return pred

        # TODO CDS model
        if isinstance(test_model, Functional):
            pred = test_model.predict(
                [np.array([first_embeddings[0]]),
                 np.array([POS.get_pos_vector(correlation['w1_pos'])]),
                 np.array([first_embeddings[1]]),
                 np.array([POS.get_pos_vector(correlation['w2_pos'])]),
                 np.array([POS.get_pos_vector(correlation['target_pos'])])
                 ])
            return pred


    def spearman_correlation_model_predictions_and_oracle(self, test_model, evaluator: SimilarityEvaluator,
                                                          save_on_file, path, mode,
                                                          pretrained_embeddings_model: KeyedVectors):

        similarities = self.collect_similarity_of_predictions(test_model, evaluator, save_on_file, path, mode,
                                                              pretrained_embeddings_model)
        return spearmanr([similarities[key] for key in similarities], [self.oracle.correlations[key]['value']
                                                                       for key in self.oracle.correlations])


def oov_similarity_sister_terms(seed, measure, model, model_name, pretrained_embeddings_model):
    reader = OOVSisterTerms_LineReader()
    base_dir = 'data/similarity_pedersen_test/oov_sister_terms_with_definitions/seed_' + seed
    oracle_path = os.path.join(base_dir, 'oov_oracle_' + measure + '.txt')

    oracle = OOVSisterTerms_POSAwareOracle(path=oracle_path)
    oracle.collect_correlations(reader)

    oracle.remove_correlations_with_oov(
        Checker.get_instance_from_path('data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True))

    evaluator = SimilarityEvaluator('cosine_similarity')
    tester = OOVSisterTerms_POSAwareTester(oracle)

    output_dir = os.path.join(base_dir, model_name + '_model')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'oov_oracle_' + measure + '_results.txt')

    spearman = tester.spearman_correlation_model_predictions_and_oracle(test_model=model,
                                                                        evaluator=evaluator,
                                                                        save_on_file=True, path=output_path,
                                                                        mode='w+',
                                                                        pretrained_embeddings_model=pretrained_embeddings_model)
    print('\t'.join([seed, model_name, measure, str(spearman.correlation)]))


# TODO you can join this and in voc method?
def collect_test_of_size(n_test, test_size, k_clusters: dict, ouput_path=None):
    tests = []
    exists_available = True
    i = 0

    while i in range(0, n_test) and exists_available:
        test = []
        available_centers = [center for center in k_clusters if len(k_clusters[center]) != 0]

        for j in range(0, test_size):
            if j + len(available_centers) < test_size:
                print('la len del test finora + quella dei centri ancora disponibili e\' minore della size del test')
                print(str(j + len(available_centers)) + ' vs ' + str(test_size))
                exists_available = False
                break

            center = random.choice(available_centers)
            available_centers.remove(center)

            d = random.choice(k_clusters[center])
            test.append(d)
            k_clusters[center].remove(d)

        if len(test) == test_size:
            test.sort()
            tests.append(test)
        else:
            break
        i += 1

    if ouput_path is not None:
        with open(ouput_path, 'w+') as output:
            output.write('\t'.join(['TEST_N', 'oov', 'synset_oov', 'first', 'second', 'synset_second', 'target_pos',
                                    'w1_pos', 'w2_pos', 'value', '#\n']))
            for i in range(0, len(tests)):
                for d in tests[i]:
                    (similarity_value, couple) = d
                    couple: SynsetOOVCouple = couple
                    output.write('\t'.join(
                        [str(i + 1), couple.oov, couple.synset_oov.name(), couple.first[0] + ' ' + couple.first[1],
                         couple.second, couple.synset_second.name(), couple.target_pos,
                         couple.w1_pos, couple.w2_pos,
                         str(similarity_value), '#\n']))
    return tests


def micro_lists_oov_pedersen_similarity(model, pretrained_embeddings_model, root_data_model, destination_dir,
                                        similarities_function_names=None, model_name=None, seed=None):
    if similarities_function_names is None:
        similarities_function_names = ['path', 'lch', 'wup', 'res', 'jcn', 'lin']
    spearman = {}
    evaluator = SimilarityEvaluator('cosine_similarity')

    root_data_model = root_data_model + '/'
    destination_dir = destination_dir + '/'

    K = 15
    N_TEST = 500
    TEST_SIZE = 6
    checker = Checker.get_instance_from_path('data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
    checker.model = pretrained_embeddings_model
    for measure in similarities_function_names:
        n_couple_clusters = retrieve_oov_couples_divided_by_value_of_similarity(
            input_path=root_data_model + '/oov_oracle_' + measure + '.txt')

        """print('-----------------------------')
        lengths_sublists = [(value, len(n_couple_clusters[value])) for value in n_couple_clusters]
        print(lengths_sublists)"""
        """save_clusters(lists=lengths_sublists,
                      output_path=root_data_model + 'in_vocabulary_similarities/seed_' + seed + '_clusters_n_' + measure + '.txt')
        """"""min_len = min([len for (value, len) in lengths_sublists])
        max_len = max([len for (value, len) in lengths_sublists])
        avg = mean([len for (value, len) in lengths_sublists])
        print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
        print('\n')"""

        k_clusters = ClusterMinDiam.k_clusters_of_min_diameter(k=K, n_clusters=n_couple_clusters)

        """lengths_sublists = [(value, len(k_clusters[value])) for value in k_clusters]
        print(lengths_sublists)
        print('-----------------------------')"""
        """save_clusters(lists=lengths_sublists,
                      output_path=root_data_model + 'in_vocabulary_similarities/seed_' + seed + '_clusters_k_' + measure + '.txt')
        """"""min_len = min([len for (value, len) in lengths_sublists])
                    max_len = max([len for (value, len) in lengths_sublists])
                    avg = mean([len for (value, len) in lengths_sublists])
                    print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
                    print('----------------')"""

        tests = collect_test_of_size(n_test=N_TEST, test_size=TEST_SIZE, k_clusters=k_clusters,
                                     ouput_path=os.path.join(root_data_model, destination_dir,
                                                             measure + '_micro_lists_test.txt'))

        """print('----------------')
        lengths_sublists = [len(test) for test in tests]
        min_len = min(lengths_sublists)
        max_len = max(lengths_sublists)
        avg = mean(lengths_sublists)
        print(lengths_sublists)
        print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
        print('----------------')"""
        spearman[measure] = []
        for i in range(0, len(tests)):
            oracle = OOVSisterTerms_POSAwareOracle(path=None)
            for d in tests[i]:
                (similarity_value, couple) = d
                couple: SynsetOOVCouple = couple
                oracle.add_correlations(value=similarity_value, oov=couple.oov,
                                        synset_oov=couple.synset_oov.name(), first=couple.first,
                                        second=couple.second, synset_second=couple.synset_second.name(),
                                        target_pos=couple.target_pos, w1_pos=couple.w1_pos, w2_pos=couple.w2_pos)

            oracle.remove_correlations_with_oov(checker)

            tester: OOVSisterTerms_POSAwareTester = OOVSisterTerms_POSAwareTester(oracle)
            output_path = os.path.join(root_data_model, destination_dir, measure + '_output_micro_lists_test.txt')

            spearman[measure].append(
                tester.spearman_correlation_model_predictions_and_oracle(model, evaluator, save_on_file=True,
                                                                         path=output_path,
                                                                         mode='a+',
                                                                         pretrained_embeddings_model=pretrained_embeddings_model)
            )

        distribution = Gauss(data=[-x.correlation for x in spearman[measure]])
        distribution.save(output_path=os.path.join(root_data_model, destination_dir, measure + '_gauss_test.png'),
                          title=f"{measure} mini-lists spearman results")
        print('\t'.join([seed, model_name, measure, str(distribution.mu), str(distribution.std)]))
