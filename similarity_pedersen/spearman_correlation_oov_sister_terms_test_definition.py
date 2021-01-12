import tensorflow as tf
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import csv
import os

from similarity_pedersen.collect_pedersen_similarities import *
from base_model.BaselineAdditiveModel import BaselineAdditiveModel
from preprocessing.w2v_preprocessing_embedding import PreprocessingWord2VecEmbedding, OOVWordException
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
    def __init__(self, positive_input_path, negative_input_path):
        self.positive_input_path = positive_input_path
        self.negative_input_path = negative_input_path

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
        return positive_couples + negative_couples


class OOVSisterTerms_LineReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[13])
            oov = line[1]
            synset_oov = line[5]
            first = line[2:4]
            second = line[10]
            synset_second = line[9]
            target_pos = line[6]
            w1_pos = line[7]
            w2_pos = line[8]

            return value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos
        except ValueError:
            raise UnexpectedValueInLine(line)

#TODO
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

    def collect_correlations(self, reader: OOVSisterTerms_LineReader, index_range: range):
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

