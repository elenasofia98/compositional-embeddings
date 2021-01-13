import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors, FastText
from gensim.models.keyedvectors import FastTextKeyedVectors
from scipy.stats import spearmanr
import os

from base_model.ParentModel import ParentModel
from similarity_pedersen.collect_pedersen_similarities import *
from base_model.BaselineAdditiveModel import BaselineAdditiveModel
from utility_test.oracle.oracle import POSAwareOracle, Oracle, POSAwareOOVOracle
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from utility_test.tester.tester import Tester, TestWriter, LineReader, UnexpectedValueInLine
from word_in_vocabulary import WordInSynset
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
                    oov_s, w2_s2 = WordInSynset(word=split[2], synset_name=split[0], pos=split[4]), WordInSynset(word=split[3], synset_name=split[1], pos=split[4])
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

                        #TODO you can omit controls like these cause the lists should be ordered in the same way, check if there are issue why this could be not true (errors, exceptions ecc)
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
                    value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos = reader.readline(line)
                    self.add_correlations(value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos)
                except UnexpectedValueInLine:
                    continue

    def remove_correlations_with_oov(self, checker: Checker):
        del_keys = []
        for key in self.correlations:
            correlation = self.correlations[key]
            if not checker.is_in_vocabulary(correlation['first'][0]) or not checker.is_in_vocabulary(correlation['first'][1]) or not checker.is_in_vocabulary(correlation['second']):
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
            header = '\t'.join(['id', 'oov', 'synset_oov', 'first', 'second', 'synset_second', 'target_pos', 'w1_pos', 'w2_pos', 'oracle_value', 'model_value' '#\n'])
            writer = TestWriter(path, header, mode)

        for i in self.oracle.correlations:
            correlation = self.oracle.correlations[i]

            prediction_1 = self._predict_according_to(test_model, pretrained_embeddings_model, correlation)
            prediction_2 = pretrained_embeddings_model.word_vec(correlation['second'])

            similarities[i] = evaluator.similarity_function(np.array(prediction_1), np.array(prediction_2))

            if save_on_file:
                writer.write_free_line(index=i, line=[correlation['oov'], correlation['synset_oov'],
                                                      '['+ correlation['first'][0] + ' ' + correlation['first'][1]+']',
                                                      correlation['second'], correlation['synset_second'],
                                                      correlation['target_pos'], correlation['w1_pos'], correlation['w2_pos'],
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

        #TODO mine model
        """#mine model
        first_embeddings = np.array([pretrained_embeddings_model.word_vec(word) for word in correlation['first']])"""

    def spearman_correlation_model_predictions_and_oracle(self, test_model, evaluator: SimilarityEvaluator,
                                                          save_on_file, path, mode,
                                                          pretrained_embeddings_model: KeyedVectors):

        similarities = self.collect_similarity_of_predictions(test_model, evaluator, save_on_file, path, mode, pretrained_embeddings_model)
        return spearmanr([similarities[key] for key in similarities], [self.oracle.correlations[key]['value']
                                                                       for key in self.oracle.correlations])


def oov_similarity_sister_terms(seed, measure, model, model_name, pretrained_embeddings_model):
    reader = OOVSisterTerms_LineReader()
    base_dir = 'data/similarity_pedersen_test/oov_sister_terms_with_definitions/seed_' + seed
    oracle_path = os.path.join( base_dir, 'oov_oracle_' + measure + '.txt')

    oracle = OOVSisterTerms_POSAwareOracle(path=oracle_path)
    oracle.collect_correlations(reader)

    oracle.remove_correlations_with_oov(
        Checker.get_instance_from_path('data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True))

    evaluator = SimilarityEvaluator('cosine_similarity')
    tester = OOVSisterTerms_POSAwareTester(oracle)

    output_dir = os.path.join(base_dir, model_name+'_model')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, 'oov_oracle_' + measure + '_results.txt')

    spearman = tester.spearman_correlation_model_predictions_and_oracle(test_model=model,
                                                                        evaluator=evaluator,
                                                                        save_on_file=True, path=output_path,
                                                                        mode='w+',
                                                                        pretrained_embeddings_model=pretrained_embeddings_model)
    print('\t'.join([seed, model_name, measure, str(spearman.correlation)]))
