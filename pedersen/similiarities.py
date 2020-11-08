from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

from enum import Enum

class SimilarityFunction(Enum):
    path = wn.path_similarity
    lch = wn.lch_similarity
    wup = wn.wup_similarity


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
