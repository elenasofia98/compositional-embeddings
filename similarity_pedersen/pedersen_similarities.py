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


class SynsetCouple:
    def __init__(self, s1: Synset, w1, s2: Synset, w2, s_pos):
        self.s1 = s1
        self.w1 = w1
        self.s2 = s2
        self.w2 = w2
        self.s_pos = s_pos


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
    def read(input_path):
        couples = []
        first = True
        with open(input_path, 'r') as input:
            while True:
                line = input.readline()
                if not line:
                    return couples

                if first:
                    first = False
                    continue

                split = line.split('\t')
                couples.append(SynsetCouple(s1=wn.synset(split[0]), w1=split[2],
                                            s2=wn.synset(split[1]), w2=split[3], s_pos=split[4]))


class Comparator:
    def __init__(self, couples: list, similarity_function):
        self.couples = couples
        self.similarity_function = similarity_function

    def write_similarities(self, path, header):
        output = open(path, 'w+')
        output.write(header)
        output.writelines(self.get_similarities())
        output.close()

    def get_similarities(self):
        similarities = []
        for couple in self.couples:
            if type(couple) is SynsetCouple:
                similarities.append(
                    '\t'.join([couple.s1.name(), couple.s2.name(), couple.w1, couple.w2,
                               str(self.similarity_function(couple.s1, couple.s2)), couple.s_pos, '#\n']))
            else:
                raise ValueError
        return similarities
