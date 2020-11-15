from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from enum import Enum
from nltk.corpus import wordnet_ic
import nltk
nltk.download('wordnet_ic')


class SimilarityFunction(Enum):
    path = wn.path_similarity
    lch = wn.lch_similarity
    wup = wn.wup_similarity

    res = lambda x, y: wn.res_similarity(x, y, wordnet_ic.ic('ic-brown.dat'))
    jcn = lambda x, y: wn.jcn_similarity(x, y, wordnet_ic.ic('ic-brown.dat'))
    lin = lambda x, y: wn.lin_similarity(x, y, wordnet_ic.ic('ic-brown.dat'))


class SynsetCouple:
    def __init__(self, s1: Synset, w1, s2: Synset, w2):
        self.s1 = s1
        self.w1 = w1
        self.s2 = s2
        self.w2 = w2


class Comparator:
    def __init__(self, couples: list, similarity_function):
        self.couples = couples
        self.similarity_function = similarity_function

    def write_similarities(self, path):
        output = open(path, 'w')
        header = '\t'.join(['S_OOV', 'S2', 'OOV',  'W2', 'SIMILARITY', '\n'])
        output.write(header)
        output.writelines(self.get_similarities())
        output.close()

    def get_similarities(self):
        similarities = []
        for couple in self.couples:
            if type(couple) is SynsetCouple:
                similarities.append('\t'.join([couple.s1.name(), couple.s2.name(),
                                    couple.w1, couple.w2, str(self.similarity_function(couple.s1, couple.s2)), '#\n']))
            else:
                raise ValueError
        return similarities
