from pedersen.similiarities import SynsetCouple, Comparator, SimilarityFunction
from word_in_vocabulary import WNManager, Checker
from nltk.corpus import wordnet as wn
import random


def pick_from(s1, wn_manager: WNManager, checker: Checker ):
    syns = s1.hypernyms() + s1.hyponyms() + s1.instance_hyponyms()

    for i in range(0, len(syns)):
        s2 = random.choice(syns)
        words = [x for x in s2.lemma_names()
                 if not wn_manager.is_expression(x) and checker.is_in_vocabulary(x)]
        if len(words) != 0:
            w2 = random.choice(words)
            return s2, w2

    return None, None


def oov_similarity(similarity_function, path):
    wn_manager = WNManager()
    checkerGoogleNews = Checker('../data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    oovs = checkerGoogleNews.get_OOV(wn_manager.lemma_from_synsets(allow_expression=False))
    couples = []
    for oov in oovs:
        for s1 in wn.synsets(oov):
            if s1.pos() == 'n' or s1.pos() == 'v':
                s2, w2 = pick_from(s1, wn_manager, checkerGoogleNews)
                if s2 is not None:
                    couples.append(SynsetCouple(s1, oov, s2, w2))

    comparator = Comparator(couples, similarity_function)
    comparator.write_similarities(path)

    """oov_pedersen = open('oov_pedersen.txt', 'w+')
    for couple in couples:
        oov_pedersen.write('\t'.join([couple.s1.name(), couple.w1, '\n']))
    oov_pedersen.close()"""
    

oov_similarity(SimilarityFunction.lch, path='lch_oov.txt')