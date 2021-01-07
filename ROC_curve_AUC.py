import random

import sklearn as sklearn
from gensim.models import KeyedVectors, FastText
from matplotlib import pyplot

from similarity_pedersen.collect_pedersen_similarities import positive_negative_couples_from
from similarity_pedersen.similiarities import SynsetCouple
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator


def get_examples(model, positive_couples_examples, negative_couples_examples, pos=None):
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
    print('len positives:', len(positive_couples_examples), 'len negatives:', len(negative_couples_examples))

    positives = 0
    negatives = 0

    for couple in positive_couples_examples:
        #couple: SynsetCouple = couple
        if pos is None or couple.s_pos == pos:
            y_true.append(POSITIVE)
            pred = - evaluator.similarity_function(model.word_vec(couple.w1), model.word_vec(couple.w2))
            #pred = model.similarity(couple.w1, couple.w2)
            probas_pred.append(pred)

            positives += 1

    for couple in negative_couples_examples:
        #couple: SynsetCouple = couple
        if pos is None or couple.s_pos == pos:
            y_true.append(NEGATIVE)
            pred = - evaluator.similarity_function(model.word_vec(couple.w1), model.word_vec(couple.w2))
            #pred = model.similarity(couple.w1, couple.w2)
            probas_pred.append(pred)

            negatives += 1

    print(f'positives, negatives in resulting lists: {positives}, {negatives}')
    return y_true, probas_pred



pretrained_embeddings_path = 'data/pretrained_embeddings/cc.en.300.bin'
model: FastText = (FastText.load_fasttext_format(pretrained_embeddings_path)).wv

seeds = ['19', '99', '200', '1999', '5348']
base = 'data/similarity_pedersen_test/_fasttext/sister_terms_vocabulary_only/ROC_curves_fasttext_model/seed_'
names = {'n': '_noun_only', 'v': '_verbs_only', None: ''}

for seed in seeds:
    positive_couples, negative_couples = positive_negative_couples_from(
        positive_input_path='data/similarity_pedersen_test/_fasttext/sister_terms_vocabulary_only/seed_' + seed + '/in_voc_sister_terms_positive.txt',
        negative_input_path='data/similarity_pedersen_test/_fasttext/sister_terms_vocabulary_only/seed_' + seed + '/in_voc_sister_terms_negative.txt')

    for pos in names:
        y_true, probas_pred = get_examples(model, positive_couples, negative_couples, pos=pos)
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, probas_pred)
        auc = sklearn.metrics.auc(recall, precision)
        print(seed, pos, auc)

        pyplot.plot(recall, precision)
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.title('Precision-Recall ROC curve')
        pyplot.suptitle('AUC=' + str(auc))
        # show the plot
        path = base + seed + '_wiki_fasttext_precision_recall_curve' + names[pos] + '.png'
        pyplot.savefig(fname=path)
        pyplot.close()

