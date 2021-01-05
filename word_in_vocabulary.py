from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from nltk.corpus import wordnet as wn
import os
from enum import Enum

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.wrappers import FastText


class PretrainedEmbeddingModel(Enum):
    w2v = 0
    glove = 1
    fasttext = 2


class Checker:
    name_to_type_map = {'glove.6B.300d.txt': PretrainedEmbeddingModel.glove,
                        'GoogleNews-vectors-negative300.bin': PretrainedEmbeddingModel.w2v,
                        'enwiki_20180420_300d.txt': PretrainedEmbeddingModel.w2v,
                        'cc.en.300.bin': PretrainedEmbeddingModel.fasttext}

    def get_OOV(self, test_words: list):
        oov = {}
        for word in test_words:
            if not self.is_in_vocabulary(word):
                oov[word] = 1
        return oov.keys()

    def get_vocabulary(self, test_words: list):
        in_voc = {}
        for word in test_words:
            if self.is_in_vocabulary(word):
                in_voc[word] = 1
        return in_voc.keys()

    def is_in_vocabulary(self, word):
        raise NotImplementedError("Not supported. Choose a subclass and instantiate an object of it")

    @staticmethod
    def name_to_type(name_to_type_map):
        for name in name_to_type_map:
            if name_to_type_map[name] not in [item.value for item in PretrainedEmbeddingModel]:
                raise KeyError("Can't map this value to any of known PretrainedEmbeddingModel. Add item to it.")
        Checker.name_to_type_map = name_to_type_map


    @staticmethod
    def get_instance_from_path(path, binary=None):
        name = os.path.basename(path)
        if Checker.name_to_type_map[name] == PretrainedEmbeddingModel.w2v:
            return W2VChecker(path, binary)

        if Checker.name_to_type_map[name] == PretrainedEmbeddingModel.glove:
            return GloveChecker(path, binary)

        if Checker.name_to_type_map[name] == PretrainedEmbeddingModel.fasttext:
            return FastTextChecker(path)

        raise NotImplementedError("Not supported. Static field name_to_type_map must be customized")


class KeyedVectorChecker(Checker):
    def __init__(self, pretrained_embeddings_path, binary: bool):
        self.model = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=binary)
        # self.model.init_sims(replace=True)

    def is_in_vocabulary(self, word):
        try:
            self.model.word_vec(word)
            return True
        except KeyError:
            return False


class FastTextChecker(Checker):
    #TODO load_word2vec_format (KeyedVectors) come funziona? perche' ora il modello e' sottoclasse, dovrebbe gestire != gli OOV!
    def __init__(self, pretrained_embeddings_path):
        self.model: FastTextKeyedVectors = (FastText.load_fasttext_format(pretrained_embeddings_path)).wv

    # TODO decidi se vuoi sapere come sta computando quella roba anche se torna true
    def is_in_vocabulary(self, word):
        try:
            self.model.word_vec(word)
            return True
        except KeyError:
            # dovrebbe tornare false solo quando nessuno degli n-gram e' contenuto nel vocabolario
            return False


class W2VChecker(KeyedVectorChecker):
    def __init__(self, pretrained_embeddings_path, binary: bool):
        super(W2VChecker, self).__init__(pretrained_embeddings_path, binary)


class GloveChecker(KeyedVectorChecker):
    def __init__(self, pretrained_embeddings_path, binary):
        new_pretrained_embeddinds_path = os.path.split(pretrained_embeddings_path)[0] + "/gensim_glove_vectors.txt"

        if not os.path.exists(new_pretrained_embeddinds_path):
            glove2word2vec(glove_input_file=pretrained_embeddings_path,
                           word2vec_output_file=new_pretrained_embeddinds_path)

        super(GloveChecker, self).__init__(new_pretrained_embeddinds_path, binary=binary)


class WNManager:
    def __init__(self):
        self.all_synsets = wn.all_synsets

    @staticmethod
    def is_expression(lemma: str):
        if lemma.find('_') == -1:
            return False
        else:
            return True

    def lemma_from_synsets(self, allow_expression: bool):
        wn_lemmas = {}
        for ss in self.all_synsets():
            for lemma in ss.lemma_names():
                if allow_expression or not WNManager.is_expression(lemma):
                    wn_lemmas[lemma] = 0
        return list(wn_lemmas)


def found_oov():
    wn_manager = WNManager()

    pretrained_embeddinds_path = 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin'
    checkerGoogleNews = W2VChecker(pretrained_embeddinds_path, binary=True)

    words = wn_manager.lemma_from_synsets(allow_expression=False)

    oovs = checkerGoogleNews.get_OOV(words)
    writeOOVS(oovs, "data\oov\oov_google.txt")
    print(len(words))
    print((len(oovs) / len(words)) * 100)
    print(f'OOVs GoogleNews size (expressions NOT allowed): {len(oovs)}')

    """
    oovs = checkerGoogleNews.get_OOV(wn_manager.lemma_from_synsets(allow_expression=True))
    print(f'OOVs GoogleNews size (expressions allowed): {len(oovs)}')
    """

    pretrained_embeddinds_path = 'data/pretrained_embeddings/enwiki_20180420_300d.txt'
    checkerWikipedia = W2VChecker(pretrained_embeddinds_path, binary=False)

    oovs = checkerWikipedia.get_OOV(words)
    writeOOVS(oovs, "data\oov\oov_wikipedia.txt")
    print(len(words))
    print((len(oovs) / len(words)) * 100)
    print(f'OOVs Wikipedia size (expressions NOT allowed): {len(oovs)}')

    """
    oovs = checkerWikipedia.get_OOV(wn_manager.lemma_from_synsets(allow_expression=True))
    print(f'OOVs Wikipedia size (expressions allowed): {len(oovs)}')
    """


def writeOOVS(oovs: list, path: str):
    f = open(path, "a")
    for word in oovs:
        f.write(word + "\n")
    f.close()
