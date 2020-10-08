from gensim.models import KeyedVectors

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
the EmbeddingLayer provides embedding for a single word of for group of words:
Getters raise exceptions defined before
"""


class PreprocessingWord2VecEmbedding:
    def __init__(self, pretrained_embeddinds_path: str, binary: bool):
        self.model = KeyedVectors.load_word2vec_format(pretrained_embeddinds_path, binary=binary)
        self.model.init_sims(replace=True)

    def get_vector(self, word: str):
        try:
            return self.model.word_vec(word)
        except KeyError:
            raise OOVWordException(word)

    def get_vector_example(self, words):
        try:
            print(words)
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
