import os
import random

import sklearn as sklearn
from gensim.models import KeyedVectors, FastText
from matplotlib import pyplot

from base_model.utility import predict_according_to, predict_according_to_oov_couple, couple_model_pretrained_given
from similarity_pedersen.collect_pedersen_similarities import positive_negative_in_voc_synset_couples_from, \
    positive_negative_oov_synset_couples_from
from similarity_pedersen.pedersen_similarities import SynsetCouple, SynsetOOVCouple
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator


def print_plot(recall, precision, auc, output_path):
    pyplot.axis(xmin=0.0, xmax=1.05, ymin=0.0, ymax=1.05)
    pyplot.plot(recall, precision)
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.title('Precision-Recall ROC curve')
    pyplot.suptitle('AUC=' + str(auc))
    # show the plot
    pyplot.savefig(fname=output_path)
    pyplot.close()


def get_in_voc_examples(model, positive_couples_examples, negative_couples_examples, pos=None):
    POSITIVE = 1
    NEGATIVE = 0
    evaluator = SimilarityEvaluator('cosine_similarity')

    if len(positive_couples_examples) < len(negative_couples_examples):
        r = len(positive_couples_examples) / len(negative_couples_examples)
        negative_couples_examples = [c for c in negative_couples_examples if random.uniform(0, 1) <= r]
    else:
        r = len(negative_couples_examples) / len(positive_couples_examples)
        positive_couples_examples = [c for c in positive_couples_examples if random.uniform(0, 1) <= r]

    y_true = []
    probas_pred = []
    # print('len positives:', len(positive_couples_examples), 'len negatives:', len(negative_couples_examples))

    positives = 0
    negatives = 0

    for couple in positive_couples_examples:
        # couple: SynsetCouple = couple
        if pos is None or couple.s_pos == pos:
            y_true.append(POSITIVE)
            pred = - evaluator.similarity_function(model.word_vec(couple.w1), model.word_vec(couple.w2))
            # pred = model.similarity(couple.w1, couple.w2)
            probas_pred.append(pred)

            positives += 1

    for couple in negative_couples_examples:
        # couple: SynsetCouple = couple
        if pos is None or couple.s_pos == pos:
            y_true.append(NEGATIVE)
            pred = - evaluator.similarity_function(model.word_vec(couple.w1), model.word_vec(couple.w2))
            # pred = model.similarity(couple.w1, couple.w2)
            probas_pred.append(pred)

            negatives += 1

    # print(f'positives, negatives in resulting lists: {positives}, {negatives}')
    return y_true, probas_pred


def calculate_in_voc_roc_auc_precision_recall(model, model_name, base_path, seeds=None):
    names = {'n': '_noun_only', 'v': '_verbs_only', None: ''}
    if seeds is None:
        seeds = ['19', '99', '200', '1999', '5348']

    for seed in seeds:
        positive_couples, negative_couples = positive_negative_in_voc_synset_couples_from(
            positive_input_path=os.path.join(base_path, 'seed_' + seed, 'in_voc_sister_terms_positive.txt'),
            negative_input_path=os.path.join(base_path, 'seed_' + seed, 'in_voc_sister_terms_negative.txt'))

        for pos in names:
            y_true, probas_pred = get_in_voc_examples(model, positive_couples, negative_couples, pos=pos)
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, probas_pred)
            auc = sklearn.metrics.auc(recall, precision)
            if pos is not None:
                print('\t'.join([seed, pos, str(auc)]))
            else:
                print('\t'.join([seed, 'None', str(auc)]))

            dirs = os.path.join(base_path, 'ROC_curves_' + model_name + '_model')
            if not os.path.exists(dirs):
                os.mkdir(dirs)
            output_path = os.path.join(dirs,
                                       'seed_' + seed + '_' + model_name +
                                       '_precision_recall_curve' + names[pos] + '.png')

            print_plot(recall, precision, auc, output_path=output_path)


def get_oov_examples(test_model, pretrained_embeddings_model, positive_couples_examples, negative_couples_examples, pos=None):
    evaluator = SimilarityEvaluator('cosine_similarity')

    probas_pred_pos = []
    probas_pred_neg = []

    positives = 0
    negatives = 0
    err = 0

    for couple in positive_couples_examples:
        couple: SynsetOOVCouple = couple
        if pos is None or couple.target_pos == pos.upper():

            try:
                prediction_1 = predict_according_to_oov_couple(test_model, pretrained_embeddings_model, oov_couple=couple)
                prediction_2 = pretrained_embeddings_model.word_vec(couple.second)

                pred = - evaluator.similarity_function(prediction_1, prediction_2)
                probas_pred_pos.append(pred)

                positives += 1
            except KeyError:
                err += 1
                pass

    for couple in negative_couples_examples:
        couple: SynsetOOVCouple = couple
        if pos is None or couple.target_pos == pos.upper():
            try:
                prediction_1 = predict_according_to_oov_couple(test_model, pretrained_embeddings_model, oov_couple=couple)
                prediction_2 = pretrained_embeddings_model.word_vec(couple.second)

                pred = - evaluator.similarity_function(prediction_1, prediction_2)
                probas_pred_neg.append(pred)

                negatives += 1
            except KeyError:
                err += 1
                pass
    #print(f'positives, negatives, err in lists: {positives}, {negatives}, {err}')

    if len(probas_pred_pos) < len(probas_pred_neg):
        r = len(probas_pred_pos) / len(probas_pred_neg)
        probas_pred_neg = [c for c in probas_pred_neg if random.uniform(0, 1) <= r]
    else:
        r = len(probas_pred_neg) / len(probas_pred_pos)
        probas_pred_pos = [c for c in probas_pred_neg if random.uniform(0, 1) <= r]


    #print(f'positives, negatives in resulting lists: {len(probas_pred_pos)}, {len(probas_pred_neg)}')

    POSITIVE = 1
    NEGATIVE = 0
    return [POSITIVE for p in probas_pred_pos] + [NEGATIVE for n in probas_pred_neg], probas_pred_pos + probas_pred_neg


def calculate_oov_roc_auc_precision_recall(test_model, pretrained_embeddings_model, model_name, base_path, seeds=None):
    names = {'n': '_noun_only', 'v': '_verbs_only', None: ''}
    if seeds is None:
        seeds = ['19', '99', '200', '1999', '5348']

    for seed in seeds:
        seed_path = os.path.join(base_path, 'seed_'+seed)
        base_output_path = os.path.join(seed_path, model_name+'_model', 'ROC_curves')
        if not os.path.exists(base_output_path):
            os.mkdir(base_output_path)

        positive_couples, negative_couples = positive_negative_oov_synset_couples_from(
            positive_input_path=os.path.join(seed_path, 'oov_definition_sister_terms_positive.txt'),
            negative_input_path=os.path.join(seed_path, 'oov_definition_sister_terms_negative.txt'))

        for pos in names:
            y_true, probas_pred = get_oov_examples(test_model, pretrained_embeddings_model,
                                                   positive_couples, negative_couples, pos=pos)
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, probas_pred)
            auc = sklearn.metrics.auc(recall, precision)
            if pos is not None:
                print('\t'.join([model_name, seed, pos, str(auc)]))
            else:
                print('\t'.join([model_name, seed, 'n+v', str(auc)]))

            output_path = os.path.join(base_output_path, model_name + '_precision_recall_curve' + names[pos] + '.png')

            print_plot(recall, precision, auc, output_path=output_path)


#TODO fast text model with only oov
"""
pretrained_path = 'data/pretrained_embeddings/cc.en.300.bin'
#model = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)

model = FastText.load_fasttext_format(pretrained_path)
model = model.wv

model_name = 'wiki_fasttext'
base_path = 'data/similarity_pedersen_test/_fasttext/sister_terms'

calculate_in_voc_roc_auc_precision_recall(model, model_name, base_path)
"""
"""
'additive': ['additive_model', 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', True],
        'parent': ['parent_model', 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', True],
        'fasttext': ['fasttext_model', 'data/pretrained_embeddings/cc.en.300.bin', None],
"""
base_path = 'data/similarity_pedersen_test/oov_sister_terms_with_definitions'
model_mappings = {

        'functional': ['functional_model', 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', True],
        'functional_no_test': ['functional_no_test_model', 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin', True]
    }

for model_name in model_mappings:
    test_model, pretrained_embeddings_model = couple_model_pretrained_given(model_name, model_mappings)
    calculate_oov_roc_auc_precision_recall(test_model, pretrained_embeddings_model, model_name, base_path, seeds=None)