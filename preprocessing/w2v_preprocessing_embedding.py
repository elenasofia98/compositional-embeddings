from gensim.models import KeyedVectors
from enum import Enum
import numpy

"""
Custom Exception for out of vocabulary (oov) word: keep track of that word
"""


class OOVWordException(Exception):
    def __init__(self, word):
        message = "Invalid Word: word \'" + word + "\' is out of this vocabulary"
        super().__init__(message)
        self.message = message
        self.word = word


"""
Custom Exception for examples that contains oov 
"""


class OOVExampleException(Exception):
    def __init__(self):
        message = "Invalid Example: context word or words in definition are out of vocabulary"
        super().__init__(message)
        self.message = message


"""
Given file's path containing embeddings in Word2Vec format (binary and not),
the PreprocessingWord2VeEmbedding provides embedding for a single word of for group of words:
- get_vector method provides, if known, the embedding for a single word
- get_vector_example provides embeddings for a list of word: the first word is meant to be target, second and third data
- get_vector_example_couple works as get_vector_example does but returns for each list of words 2 examples:
  in this method the order in which data words are given is ignored and is computed the first example as defined before 
  and the second with data words in reverse order
- get get_list_of_vectors returns embedding for each of the given words
Getters raise exceptions defined before when one of the given word has no embedding
"""


class PreprocessingWord2VecEmbedding:
    def __init__(self, pretrained_embeddinds_path: str, binary: bool):
        self.model = KeyedVectors.load_word2vec_format(pretrained_embeddinds_path, binary=binary)
        # self.model.init_sims(replace=True)

    def get_vector(self, word: str):
        try:
            return self.model.word_vec(word)
        except KeyError:
            raise OOVWordException(word)

    def get_vector_example(self, words):
        try:
            target, data = words[0], words[1:]
            vectors_example = {'target': self.get_vector(target),
                               'data': [self.get_vector(word) for word in data]}
            return vectors_example
        except OOVWordException:
            raise OOVExampleException()

    def get_vector_example_couple(self, words):
        try:
            target, data = words[0], words[1:]
            vectors_example_0 = {'target': self.get_vector(target),
                                 'data': [self.get_vector(word) for word in data]}
            vectors_example_1 = {'target': self.get_vector(target),
                                 'data': [self.get_vector(word) for word in reversed(data)]}
            return [vectors_example_0, vectors_example_1]
        except OOVWordException:
            raise OOVExampleException()

    def get_list_of_vectors(self, words):
        vectors = [self.get_vector(word) for word in words]
        return vectors


class POS(Enum):
    VERB = 'v'
    ADJ = 'a'
    NOUN = 'n'

    @staticmethod
    def get_pos_vector(pos):
        if pos == POS.VERB:
            return [0, 0, 1]
        if pos == POS.ADJ:
            return [0, 1, 0]
        else:
            return [1, 0, 0]


class POSAwarePreprocessingWord2VecEmbedding(PreprocessingWord2VecEmbedding):
    def __init__(self, pretrained_embeddinds_path: str, binary: bool):
        super().__init__(pretrained_embeddinds_path, binary)

    def get_vector_example(self, words, pos):
        vectors_example = super().get_vector_example(words)
        vectors_example['pos'] = POS.get_pos_vector(pos)
        return vectors_example
