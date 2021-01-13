import random
from base_model.model import BadExampleException, Model
from nltk.corpus import wordnet as wn

from word_in_vocabulary import Checker, WNManager


class ParentModel(Model):
    def __init__(self, path, binary):
        self.checker = Checker.get_instance_from_path(path=path, binary=binary)
        super(ParentModel, self).__init__()

    def process_example(self, target, data):
        if len(data) != 1:
            raise BadExampleException

        super(ParentModel, self).process_example(target, data)

    def predict(self, x, pos_tag=None):
        #word_vectors = []
        """if pos_tags is None:
            pos_tags = [None for word in x]

        for i in range(0, len(x)):"""
        word = x#[i]
        pos = pos_tag#[i]
        for hyp in wn.synsets(lemma=word, pos=pos.lower())[0].hypernyms():
            in_voc = [lemma for lemma in hyp.lemma_names()
                      if self.checker.is_in_vocabulary(word=lemma) and not WNManager.is_expression(lemma=lemma)]
            if len(in_voc) > 0:
                choice = random.choice(in_voc)
                # print(choice)
                return self.checker.model.word_vec(choice)
        #print('parent not found, entity choosen pos = ', pos)
        return self.checker.model.word_vec('entity')


