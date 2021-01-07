import random

import numpy as np
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

    def predict(self, x, pos=None):
        for hyp in wn.synsets(lemma=x, pos=pos)[0].hypernyms():
            in_voc = [lemma for lemma in hyp.lemma_names()
                      if self.checker.is_in_vocabulary(word=lemma) and not WNManager.is_expression(lemma=lemma)]
            if len(in_voc) > 0:
                choice = random.choice(in_voc)
                #print(choice)
                return self.checker.model.word_vec(choice)
