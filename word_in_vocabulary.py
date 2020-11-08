from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn


class Checker:
    def __init__(self, pretrained_embeddinds_path, binary: bool):
        self.model = KeyedVectors.load_word2vec_format(pretrained_embeddinds_path, binary=binary)
        #self.model.init_sims(replace=True)

    def get_OOV(self, test_words: list):
        oov = {}
        for word in test_words:
            if not self.is_in_vocabulary(word):
                oov[word] = 1
        return oov.keys()

    def is_in_vocabulary(self, word):
        try:
            self.model.word_vec(word)
            return True
        except KeyError:
            return False


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
    checkerGoogleNews = Checker(pretrained_embeddinds_path, binary=True)

    words = wn_manager.lemma_from_synsets(allow_expression=False)

    oovs = checkerGoogleNews.get_OOV(words)
    writeOOVS(oovs, "data\oov\oov_google.txt")
    print(len(words))
    print((len(oovs)/len(words)) * 100)
    print(f'OOVs GoogleNews size (expressions NOT allowed): {len(oovs)}')

    """
    oovs = checkerGoogleNews.get_OOV(wn_manager.lemma_from_synsets(allow_expression=True))
    print(f'OOVs GoogleNews size (expressions allowed): {len(oovs)}')
    """

    pretrained_embeddinds_path = 'data/pretrained_embeddings/enwiki_20180420_300d.txt'
    checkerWikipedia = Checker(pretrained_embeddinds_path, binary=False)

    oovs = checkerWikipedia.get_OOV(words)
    writeOOVS(oovs, "data\oov\oov_wikipedia.txt")
    print(len(words))
    print((len(oovs)/len(words)) * 100)
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