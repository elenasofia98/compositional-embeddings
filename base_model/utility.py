import numpy as np
import tensorflow as tf
from gensim.models import FastText

from gensim.models.keyedvectors import FastTextKeyedVectors, KeyedVectors
from tensorflow.python.keras.engine.functional import Functional

from base_model.BaselineAdditiveModel import BaselineAdditiveModel
from base_model.ParentModel import ParentModel
from preprocessing.w2v_preprocessing_embedding import POS
from similarity_pedersen.pedersen_similarities import SynsetOOVCouple


def predict_according_to(test_model, pretrained_embeddings_model, correlation):
    if isinstance(test_model, ParentModel):
        prediction = test_model.predict(correlation['oov'], pos_tag=correlation['target_pos'].lower())
        return prediction

    if isinstance(test_model, KeyedVectors) or isinstance(test_model, FastTextKeyedVectors):
        prediction = test_model.word_vec(correlation['oov'])
        return prediction

    first_embeddings = np.array([pretrained_embeddings_model.word_vec(word) for word in correlation['first']])
    if isinstance(test_model, BaselineAdditiveModel):
        prediction = test_model.predict(first_embeddings)
        return prediction

    if isinstance(test_model, Functional):
        prediction = test_model.predict(
            [np.array([first_embeddings[0]]),
             np.array([POS.get_pos_vector(correlation['w1_pos'])]),
             np.array([first_embeddings[1]]),
             np.array([POS.get_pos_vector(correlation['w2_pos'])]),
             np.array([POS.get_pos_vector(correlation['target_pos'])])
             ])
        return prediction


def predict_according_to_oov_couple(test_model, pretrained_embeddings_model, oov_couple: SynsetOOVCouple):
    return predict_according_to(test_model, pretrained_embeddings_model, oov_couple.to_dictionary())


def couple_model_pretrained_given(model_name, model_mappings):
    if model_name == 'additive':
        model = BaselineAdditiveModel()
        pretrained_embeddings_model = KeyedVectors.load_word2vec_format(model_mappings[model_name][1],
                                                                        binary=model_mappings[model_name][2])
        return model, pretrained_embeddings_model

    if model_name == 'parent':
        model = ParentModel(model_mappings[model_name][1], model_mappings[model_name][2])
        pretrained_embeddings_model = KeyedVectors.load_word2vec_format(model_mappings[model_name][1],
                                                                        binary=model_mappings[model_name][2])
        return model, pretrained_embeddings_model

    if model_name == 'fasttext':
        model = FastText.load_fasttext_format(model_mappings[model_name][1])
        model = model.wv
        return model, model

    if model_name == 'functional':
        model = tf.keras.models.load_model('oov_functional_predictor.h5')
        pretrained_embeddings_model = KeyedVectors.load_word2vec_format(model_mappings[model_name][1],
                                                                        binary=model_mappings[model_name][2])
        return model, pretrained_embeddings_model

    if model_name == 'functional_no_test':
        model = tf.keras.models.load_model('oov_functional_predictor_no_sister_terms_test.h5')
        pretrained_embeddings_model = KeyedVectors.load_word2vec_format(model_mappings[model_name][1],
                                                                        binary=model_mappings[model_name][2])
        return model, pretrained_embeddings_model