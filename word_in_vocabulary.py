from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from nltk.corpus import wordnet as wn
import os
from enum import Enum

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.wrappers import FastText


class WordInSynset:
    def __init__(self, word, synset_name, pos):
        self.word = word
        self.synset_name = synset_name
        self.pos = pos.upper()

    @staticmethod
    def from_word_and_pos(word, pos):
        ss = wn.synsets(word, pos=pos)
        if len(ss) > 0:
            synset_name = ss[0].name()
            return WordInSynset(word, synset_name, pos)
        else:
            return None

    def equals(self, s):
       return self.synset_name == s.synset_name and self.word == s.word


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
                raise KeyError("This value isn't mapped to a known PretrainedEmbeddingModel.")
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
        return word in self.model.vocab


class FastTextChecker(Checker):
    # TODO load_word2vec_format (KeyedVectors) come funziona? Non ha gli n-grams..?
    def __init__(self, pretrained_embeddings_path):
        model = FastText.load_fasttext_format(pretrained_embeddings_path)
        self.model: FastTextKeyedVectors = model.wv

    # TODO decidi se vuoi sapere come sta computando quella roba anche se torna true
    def is_in_vocabulary(self, word):
        try:
            self.model.word_vec(word)
            return True
        except KeyError:
            # torna false solo quando nessuno degli n-gram e' contenuto nel vocabolario
            return False
        # return word in self.model.vocab


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


"""def not_done(syns, done):
    out_syns = []
    done_syns = []
    for s in syns:
        s: WordInSynset = s
        found = False
        i = 0
        while not found and i < len(done):
            #print(done[i].synset_name == s.synset_name, done[i].word == s.word)
            if s.equals(done[i]):
                found = True
                done_syns.append(s)
                break
            i += 1
        if not found:
            out_syns.append(s)

    return out_syns, done_syns"""


def find_oov_and_synset(pretrained_embeddings_path, binary=None, pos_tags=None, output_path='oov_in_synset.txt'):
    if pos_tags is None:
        pos_tags = ['n', 'v']

    checker = Checker.get_instance_from_path(pretrained_embeddings_path, binary=binary)

    wn_manager = WNManager()
    words = wn_manager.lemma_from_synsets(allow_expression=False)
    oov_list = checker.get_OOV(words)

    synsets = []
    #done_syns = []
    for oov in oov_list:
        for pos in pos_tags:
            s = WordInSynset.from_word_and_pos(oov, pos)
            if s is not None:
                synsets.append(s)

    """if done is not None:
        synsets, done_syns = not_done(synsets, done)"""
    with open(output_path, 'w+') as output:
        header = '\t'.join(['OOV', 'SYN_NAME', 'POS', '#\n'])
        output.write(header)
        for s in synsets:
            output.write('\t'.join([s.word, s.synset_name, s.pos, '#\n']))

    return synsets
