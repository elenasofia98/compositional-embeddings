from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from enum import Enum
from nltk.corpus import wordnet_ic
import nltk
nltk.download('wordnet_ic')


class InformationContent:
    INFORMATION_CONTENT = wordnet_ic.ic('ic-brown.dat')

    @staticmethod
    def set_information_content(name):
        InformationContent.INFORMATION_CONTENT = wordnet_ic.ic(name)


class SimilarityFunction(Enum):
    path = wn.path_similarity
    lch = wn.lch_similarity
    wup = wn.wup_similarity

    res = lambda x, y: wn.res_similarity(x, y, InformationContent.INFORMATION_CONTENT)
    jcn = lambda x, y: wn.jcn_similarity(x, y, InformationContent.INFORMATION_CONTENT)
    lin = lambda x, y: wn.lin_similarity(x, y, InformationContent.INFORMATION_CONTENT)

    @staticmethod
    def name(similarity_function):
        if similarity_function == SimilarityFunction.path:
            return 'path'
        if similarity_function == SimilarityFunction.lch:
            return 'lch'
        if similarity_function == SimilarityFunction.wup:
            return 'wup'
        if similarity_function == SimilarityFunction.res:
            return 'res'
        if similarity_function == SimilarityFunction.jcn:
            return 'jcn'
        if similarity_function == SimilarityFunction.lin:
            return 'lin'

    @staticmethod
    def by_name(similarity_function_name):
        if similarity_function_name == 'path':
            return SimilarityFunction.path
        if similarity_function_name == 'lch':
            return SimilarityFunction.lch
        if similarity_function_name == 'wup':
            return SimilarityFunction.wup
        if similarity_function_name == 'res':
            return SimilarityFunction.res
        if similarity_function_name == 'jcn':
            return SimilarityFunction.jcn
        if similarity_function_name == 'lin':
            return SimilarityFunction.lin


class SynsetCouple:
    def __init__(self, s1: Synset, w1, s2: Synset, w2, s_pos):
        self.s1 = s1
        self.w1 = w1
        self.s2 = s2
        self.w2 = w2
        self.s_pos = s_pos


class SynsetOOVCouple:
    def __init__(self, oov, synset_oov, first, second, synset_second, target_pos, w1_pos, w2_pos):
        self.synset_oov = wn.synset(synset_oov)
        self.oov = oov
        self.first = first
        self.synset_second = wn.synset(synset_second)
        self.second = second
        self.target_pos = target_pos
        self.w1_pos = w1_pos
        self.w2_pos = w2_pos


class SaverSynsetCouples:
    @staticmethod
    def save(couples, output_path, header):
        with open(output_path, 'w+') as output:
            output.write(header)
            for couple in couples:
                output.write('\t'.join([couple.s1.name(), couple.s2.name(),
                                    couple.w1, couple.w2, couple.s_pos, '#\n']))

    @staticmethod
    def append(couples, output_path, header):
        with open(output_path, 'a+') as output:
            output.write(header)
            for couple in couples:
                output.write('\t'.join([couple.s1.name(), couple.s2.name(),
                                    couple.w1, couple.w2, couple.s_pos, '#\n']))


class ReaderSynsetCouples:
    @staticmethod
    def read(input_path, s1_index=0, w1_index=2, s2_index=1, w2_index=3, s_pos_index=4, exclude_first=True):
        couples = []
        first = True
        with open(input_path, 'r') as input:
            while True:
                line = input.readline()
                if not line:
                    return couples

                if exclude_first and first:
                    first = False
                    continue

                split = line.split('\t')
                couples.append(SynsetCouple(s1=wn.synset(split[s1_index]), w1=split[w1_index],
                                            s2=wn.synset(split[s2_index]), w2=split[w2_index],
                                            s_pos=split[s_pos_index]))
        return couples


class ReaderSynsetOOVCouple:
    @staticmethod
    def read(input_path, s1_index=5, w1_index=1, s2_index=9, w2_index=10, first_indexes=[2, 4],
             s_pos_index=6, w1_pos=7, w2_pos=8, exclude_first=False):
        couples = []
        first = True
        with open(input_path, 'r') as input:
            while True:
                line = input.readline()
                if not line:
                    return couples

                if exclude_first and first:
                    first = False
                    continue

                split = line.split('\t')
                couples.append(SynsetOOVCouple(oov=split[w1_index], synset_oov=split[s1_index],
                                               first=split[first_indexes[0]:first_indexes[1]], second=split[w2_index],
                                               synset_second=split[s2_index], target_pos=split[s_pos_index],
                                               w1_pos=split[w1_pos], w2_pos=split[w2_pos]))
        return couples


class Comparator:
    def __init__(self, couples: list, similarity_function):
        self.couples = couples
        self.similarity_function = similarity_function

    def write_similarities(self, path, header):
        output = open(path, 'w+')
        output.write(header)
        output.writelines(self._get_string_similarities())
        output.close()

    def get_similarities(self):
        similarities = []
        for couple in self.couples:
            if type(couple) is SynsetCouple:
                similarities.append([couple.s1.name(), couple.s2.name(), couple.w1, couple.w2,
                                     str(self.similarity_function(couple.s1, couple.s2)), couple.s_pos])
            else:
                raise ValueError
        return similarities

    def _get_string_similarities(self):
        similarities = []
        for similarity in self.get_similarities():
            similarities.append('\t'.join( similarity + ['#\n']))
        return similarities
