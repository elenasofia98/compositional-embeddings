from enum import Enum

from similarity_pedersen.pedersen_similarities import SynsetCouple, Comparator, SimilarityFunction, SaverSynsetCouples, \
    ReaderSynsetCouples, SynsetOOVCouple

from utility_test.tester.tester import UnexpectedValueInLine
from word_in_vocabulary import WNManager, Checker
from nltk.corpus import wordnet as wn
import random

from writer_reader_of_examples.writer_utility import Parser


def randomchoice(list):
    SEED = 5348
    random.seed(SEED)
    return random.choice(list)


def pick_from(s1, w1, checker: Checker, similar=True):
    if similar:
        return _similar_word_to(s1, w1, checker)
    else:
        return _dissimilar_word_to(s1, w1, checker)


def _similar_word_to(s1, w1, checker: Checker):
    hypernyms = s1.hypernyms()
    if len(hypernyms) == 0:
        return None, None

    # see hypernyms_sister_term_choice file to justify this
    sister_synss = hypernyms[0].hyponyms()
    if s1 in sister_synss:
        sister_synss.remove(s1)

    if len(sister_synss) == 0:
        return None, None
    s2 = randomchoice(sister_synss)
    in_voc = [lemma for lemma in s2.lemma_names() if lemma != w1 and
              not WNManager.is_expression(lemma) and checker.is_in_vocabulary(lemma)]

    if len(in_voc) == 0:
        return None, None
    w2 = randomchoice(in_voc)
    return s2, w2
    """for i in range(0, len(sister_synss)):
        if sister_synss[i] is not s1:
            lemma = sister_synss[i].lemma_names()[0]
            if not wn_manager.is_expression(lemma) and checker.is_in_vocabulary(lemma):
                return sister_synss[i], lemma
    return None, None"""


ALL_NAMES = [x for x in wn.all_synsets('n')]
ALL_VERBS = [x for x in wn.all_synsets('v')]


# TODO You need to fix seed (?) but with high prob. s1 and s2 are unrelated in both noun and verb cases
def _dissimilar_word_to(s1, w1, checker: Checker):
    if s1.pos() == wn.NOUN:
        syns = ALL_NAMES
    else:
        syns = ALL_VERBS

    i = 0
    while i < 7:
        s2 = random.choice(syns)
        in_voc = [x for x in s2.lemma_names() if x != w1 and
                  not WNManager.is_expression(x) and checker.is_in_vocabulary(x)]
        if len(in_voc) != 0:
            w2 = random.choice(in_voc)
            return s2, w2
        i += 1

    return None, None


def _get_all_couples_from(words, checker: Checker, similar=True, output_path=None):
    couples = []
    for w1 in words:
        for s1 in wn.synsets(w1):
            if s1.pos() == 'n' or s1.pos() == 'v':
                s2, w2 = pick_from(s1, w1, checker, similar=similar)
                if s2 is not None:
                    couples.append(SynsetCouple(s1, w1, s2, w2, s1.pos()))
    if similar:
        header = '\t'.join(['S1', 'S2', 'W1', 'W2', 'S1_POS', '#\n'])
    else:
        header = '\t'.join(['S_OOV', 'S2', 'OOV', 'W2', 'S_POS', '#\n'])
    if output_path is not None:
        SaverSynsetCouples.save(couples, output_path, header)
    return couples


def get_couples_from(words, checker: Checker, similar=True, output_path=None):
    couples = []
    for w1 in words:
        for pos in ['n', 'v']:
            ss = wn.synsets(w1, pos=pos)
            if len(ss) > 0:
                s1 = ss[0]
                s2, w2 = pick_from(s1, w1, checker, similar=similar)
                if s2 is not None:
                    couples.append(SynsetCouple(s1, w1, s2, w2, s1.pos()))
    if similar:
        header = '\t'.join(['S1', 'S2', 'W1', 'W2', 'S1_POS', '#\n'])
    else:
        header = '\t'.join(['S_OOV', 'S2', 'OOV', 'W2', 'S_POS', '#\n'])
    if output_path is not None:
        SaverSynsetCouples.save(couples, output_path, header)
    return couples


def retrieve_in_voc_couples_divided_by_value_of_similarity(positive_input_path, negative_input_path, measure_name):
    similarity_function = SimilarityFunction.by_name(measure_name)
    ordered_couple = {}

    positive_couples = ReaderSynsetCouples.read(positive_input_path)
    for couple in positive_couples:
        similarity_value = similarity_function(couple.s1, couple.s2)
        if similarity_value not in ordered_couple:
            ordered_couple[similarity_value] = []
        ordered_couple[similarity_value].append(couple)

    negative_couples = ReaderSynsetCouples.read(negative_input_path)
    for couple in negative_couples:
        similarity_value = similarity_function(couple.s1, couple.s2)
        if similarity_value not in ordered_couple:
            ordered_couple[similarity_value] = []
        ordered_couple[similarity_value].append(couple)

    return ordered_couple


class OOVSisterTerms_LineReader(object):
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


def retrieve_oov_couples_divided_by_value_of_similarity(input_path):
    reader = OOVSisterTerms_LineReader()

    ordered_couples = {}
    with open(input_path, 'r+') as input_file:
        while True:
            line = input_file.readline()
            if not line:
                break
            try:
                value, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos = reader.readline(
                    line.split('\t'))

                s_oov = SynsetOOVCouple(oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos)
                if value not in ordered_couples.keys():
                    ordered_couples[value] = []

                ordered_couples[value].append(s_oov)
            except UnexpectedValueInLine:
                continue

    return ordered_couples


def positive_negative_couples_from(positive_input_path, negative_input_path):
    positive_couples = ReaderSynsetCouples.read(positive_input_path)
    negative_couples = ReaderSynsetCouples.read(negative_input_path)
    return positive_couples, negative_couples


def compare_couples(couples, similarity_function, similarity_output_path, header):
    comparator = Comparator(couples, similarity_function)
    comparator.write_similarities(similarity_output_path, header)


"""def compare_n_couples(n, couples, similarity_function, similarity_output_path, header):
    n_couples = []
    soglia = n / len(couples)
    for i in range(0, len(couples)):
        if random.uniform(0, 1) < soglia:
            n_couples.append(couples[i])

    comparator = Comparator(n_couples, similarity_function)
    comparator.write_similarities(similarity_output_path, header)"""


def oov_couples_and_similarity(similarity_function, similarity_output_path,
                               similar=True, couples_output_path=None,
                               model_path=None, binary=True):
    wn_manager = WNManager()
    if model_path is None:
        model_path = 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin'
        binary = True

    # checker = Checker('data/pretrained_embeddings/enwiki_20180420_300d.txt', binary=False)
    checker = Checker.get_instance_from_path(model_path, binary=binary)

    oovs = checker.get_OOV(wn_manager.lemma_from_synsets(allow_expression=False))
    couples = get_couples_from(oovs, checker, similar, output_path=couples_output_path)

    compare_couples(couples, similarity_function,
                    similarity_output_path, '\t'.join(['S_OOV', 'S2', 'OOV', 'W2', 'SIMILARITY', 'S_POS', '#\n']))


def voc_couples_and_similarity(similarity_function, similarity_output_path,
                               similar=True, couples_output_path=None,
                               model_path=None, binary=True):
    wn_manager = WNManager()
    if model_path is None:
        model_path = 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin'
        binary = True

    # checker = Checker('data/pretrained_embeddings/enwiki_20180420_300d.txt', binary=False)
    checker = Checker.get_instance_from_path(model_path, binary=binary)
    in_voc = checker.get_vocabulary(wn_manager.lemma_from_synsets(allow_expression=False))
    couples = get_couples_from(in_voc, checker, similar, output_path=couples_output_path)

    compare_couples(couples, similarity_function,
                    similarity_output_path, '\t'.join(['S1', 'S2', 'W1', 'W2', 'SIMILARITY', 'S1_POS', '#\n']))


"""def similarity_by_name(measure_name):
    if measure_name == 'wup':
        return SimilarityFunction.wup
    else:
        if measure_name == 'path':
            return SimilarityFunction.path
        else:
            if measure_name == 'lch':
                return SimilarityFunction.lch
            else:
                if measure_name == 'jcn':
                    return SimilarityFunction.jcn
                else:
                    if measure_name == 'lin':
                        return SimilarityFunction.lin
                    else:
                        if measure_name == 'res':
                            return SimilarityFunction.res
                        else:
                            raise NotImplementedError('Unknown measure name. SimilarityFunction must be customized')
"""


def voc_sim(measure_name, positive_output_file, negative_output_file,
            couples_output_dir='data/similarity_pedersen_test/sister_terms',
            model_path=None, binary=True):
    similarity = SimilarityFunction.by_name(measure_name)
    voc_couples_and_similarity(similarity,
                               similarity_output_path=positive_output_file,
                               similar=True,
                               couples_output_path=couples_output_dir + '/in_voc_sister_terms_positive.txt',
                               model_path=model_path,
                               binary=binary)
    voc_couples_and_similarity(similarity,
                               similarity_output_path=negative_output_file,
                               similar=False,
                               couples_output_path=couples_output_dir + '/in_voc_sister_terms_negative.txt',
                               model_path=model_path,
                               binary=binary)


def oov_sim(measure_name, positive_output_file, negative_output_file,
            couples_output_dir='data/similarity_pedersen_test/sister_terms',
            model_path=None, binary=True):
    similarity = SimilarityFunction.by_name(measure_name)
    oov_couples_and_similarity(similarity,
                               similarity_output_path=positive_output_file,
                               similar=True,
                               couples_output_path=couples_output_dir + '/oov_sister_terms_positive.txt',
                               model_path=model_path,
                               binary=binary)
    oov_couples_and_similarity(similarity,
                               similarity_output_path=negative_output_file,
                               similar=False,
                               couples_output_path=couples_output_dir + '/oov_sister_terms_negative.txt',
                               model_path=model_path,
                               binary=binary)
