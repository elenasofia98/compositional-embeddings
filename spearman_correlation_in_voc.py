import os
import random
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from statistics import mean
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from similarity_lastra_diaz.collect_lastra_diaz_similarities import Dataset
from similarity_pedersen.collect_pedersen_similarities import voc_sim, retrieve_couples_divided_by_value_of_similarity
from similarity_pedersen.similiarities import SynsetCouple
from spearman_correlation import SimilarityEvaluator, Oracle, LineReader, UnexpectedValueInLine, Tester, TestWriter
from writer.writer_utility import Parser

""""
lch [0.4365443733268208, 0.429874178346164, 0.4394192684126842, 0.4387582020723008, 0.4456564113442707, 0.4221878575291542, 0.4480933561929856]
path [0.4357298493054063, 0.4546619969272328, 0.41240142187305223, 0.4380828511105419, 0.43395238316480983, 0.4268867175139643, 0.4384571396577155]
wup [0.4446611852954406, 0.4459860891359116, 0.4535761114235438, 0.44584172730289656, 0.4380297789053809, 0.43395704333518426, 0.4521424567877747]


res [0.3169351850759179, 0.33180653655447434]
jcn [0.17671157209996688, 0.21943115215364656]
lin [0.3884127584311433, 0.35050276924034973]
lch [0.3392544411075888, 0.3170136653896707]
path [0.314853545183368, 0.3486419262602585]
wup [0.3051954662823624, 0.2765495616454321]
"""


def append(positives_input_path, negatives_input_path, output_path):
    with open(positives_input_path, 'r') as f1:
        with open(negatives_input_path, 'r') as f2:
            with open(output_path, 'w+') as output:
                positives = f1.readlines()
                negatives = f2.readlines()

                for positive in positives:
                    if random.uniform(0, 1) > 0.98:
                        output.write(positive)
                for negative in negatives:
                    if random.uniform(0, 1) > 0.98:
                        output.write(negative)


class InVocSisterTermReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[4])
            first = line[2]
            second = line[3]

            return value, first, second
        except ValueError:
            raise UnexpectedValueInLine(line)


class InVocCorrelationOracle(Oracle):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def collect_correlations(self, reader: LineReader, index_range: range):
        parser = Parser(self.path, '\t')
        with parser:
            while True:
                line = parser.get_example_from_line_next_line(index_range)
                if not line:
                    break

                try:
                    value, first, second = reader.readline(line)
                    super().add_correlations(value, first, second)
                except UnexpectedValueInLine:
                    continue


class PetersenInVocTester(Tester):
    def __init__(self, oracle: InVocCorrelationOracle):
        self.oracle = oracle

    def test_similarity_of_predictions(self, model: KeyedVectors, evaluator: SimilarityEvaluator, save_on_file=True,
                                       path='petersen_correlations.txt', mode=None):
        similarities = {}

        if save_on_file:
            header = '\t'.join(['id', 'first', 'second', 'oracle_value', 'model_value', '#\n'])
            writer = TestWriter(path, header, mode)

        for i in self.oracle.correlations:
            correlation = self.oracle.correlations[i]

            prediction_1 = model.word_vec(correlation['first'])
            prediction_2 = model.word_vec(correlation['second'])

            similarities[i] = evaluator.similarity_function(prediction_1, prediction_2).numpy()

            if save_on_file:
                writer.write_line(i, [correlation['first'], correlation['second']],
                                  [str(correlation['value']), str(similarities[i])])

        if save_on_file:
            writer.release()

        return similarities

    def spearman_correlation_model_predictions_and_oracle(self, model: KeyedVectors, evaluator: SimilarityEvaluator,
                                                          save_on_file=True,
                                                          path='petersen_correlations.txt',
                                                          mode=None):
        similarities = self.test_similarity_of_predictions(model, evaluator, save_on_file, path, mode)
        return spearmanr([similarities[key] for key in similarities], [self.oracle.correlations[key]['value']
                                                                       for key in self.oracle.correlations])


""", 'res', 'jcn', 'lin'"""

"""
Ogni volta si ricalcolano sia le coppie che il valore della loro similitudine: con il seed fissato per gli esempi postivi di dimilitudine non ha molto senso
Risultati inaffidabili a causa della ripetizione frequente di valori per la misura basaata sul path nella tassonomia di wordnet
La cosine similarity assume valori continui invece
"""


def spearman_correlation_cosine_similarity_and_path_measure_pedersen():
    evaluator = SimilarityEvaluator('cosine_similarity')
    pretrained_embeddinds_path = 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(pretrained_embeddinds_path, binary=True)

    similarities_function_names = ['path', 'lch', 'wup', 'res', 'jcn', 'lin']
    tests = {}
    for function_name in similarities_function_names:
        tests[function_name] = []

        positive_sim_examples_file = 'data/similarity_pedersen_test/in_vocabulary_similarities/positive_' + function_name + '_in_voc.txt'
        negative_sim_examples_file = 'data/similarity_pedersen_test/in_vocabulary_similarities/negative_' + function_name + '_in_voc.txt'
        voc_sim(measure_name=function_name,
                positive_output_file=positive_sim_examples_file,
                negative_output_file=negative_sim_examples_file)

        for i in range(0, 1):
            output_path = 'data/similarity_pedersen_test/in_vocabulary_similarities/' + function_name + '_in_voc_' + str(
                i) + '.txt'
            append(positives_input_path=positive_sim_examples_file,
                   negatives_input_path=negative_sim_examples_file,
                   output_path=output_path)

            oracle = InVocCorrelationOracle(output_path)
            oracle.collect_correlations(InVocSisterTermReader(), range(0, 7))

            correlations = output_path.split('.')[0] + '_output.' + output_path.split('.')[1]
            tester = PetersenInVocTester(oracle)
            spearman = tester.spearman_correlation_model_predictions_and_oracle(
                model, evaluator,
                save_on_file=True,
                path=correlations)

            # print(str(type(sequential).__name__) + ' --> ' + str(spearman_sequential))
            print('--------------')
            print(str(type(model).__name__) + ' --> ' + str(spearman))
            print('--------------')

            tests[function_name].append(- spearman.correlation)
    for key in tests:
        print(key, tests[key])


class Cluster:
    @staticmethod
    def _farest(values, centers):
        max = 0
        for value in values:
            d_value = min([abs(value - c) for c in centers])
            if d_value > max:
                max = d_value
                next = value

        return next

    @staticmethod
    #I send my regards to Pasqui! 2 approx
    def _k_clusters_of_min_diameter(k, values):
        centers = [values[0]]
        for i in range(1, k):
            next = Cluster._farest(values, centers)
            centers.append(next)

        clusters = {}
        for center in centers:
            clusters[center] = []

        for value in values:
            min = None
            for center in centers:
                d_center_value = abs(value - center)
                if min is None or d_center_value < min:
                    min = d_center_value
                    chosen_centre = center
            clusters[chosen_centre].append(value)

        return clusters

    @staticmethod
    def k_clusters_of_min_diameter(k, n_clusters: dict):
        k_clusters = {}
        if len(n_clusters.keys()) <= k:
            for center in n_clusters:
                k_clusters[center] = [(center, x) for x in n_clusters[center]]
            return k_clusters

        clusterized_values = Cluster._k_clusters_of_min_diameter(k, [x for x in n_clusters])
        for center in clusterized_values:
            k_clusters[center] = []
            for value in clusterized_values[center]:
                k_clusters[center].extend([(value, x) for x in n_clusters[value]])
        return k_clusters


def collect_test_of_size(n_test, test_size, k_clusters:dict, ouput_path=None):
    tests = []
    exists_available = True
    i = 0

    while i in range(0, n_test) and exists_available:
        test = []
        available_centers = [center for center in k_clusters if len(k_clusters[center]) != 0]

        for j in range(0, test_size):
            if j + len(available_centers) < test_size:
                print('la len del test finora + quella dei centri ancora disponibili e\' minore della size del test')
                print(str( j + len(available_centers) ) + ' vs ' + str(test_size))
                exists_available = False
                break

            center = random.choice(available_centers)
            available_centers.remove(center)

            d = random.choice(k_clusters[center])
            test.append(d)
            k_clusters[center].remove(d)

        if len(test) == test_size:
            test.sort()
            tests.append(test)
        else:
            break
        i += 1

    if ouput_path is not None:
        with open(ouput_path, 'w+') as output:
            output.write('\t'.join(['TEST_N', 'S1', 'S2', 'W1', 'W2', 'S1_POS', 'SIMILARITY', '#\n']))
            for i in range(0, len(tests)):
                for d in tests[i]:
                    (similarity_value, couple) = d
                    output.write('\t'.join([str(i+1), couple.s1.name(), couple.s2.name(),
                                            couple.w1, couple.w2, couple.s_pos, str(similarity_value), '#\n']))
    return tests


def gauss_distribution_of(data, measure, output_path):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the PDF
    xmin = min(data)
    xmax = max(data)
    x = np.linspace(xmin, xmax)

    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, color='b')

    title = f"{measure} mini-lists spearman results: mu = {mu:.2f},  std = {std:.2f}"
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

    return mu, std


def save_clusters(lists, output_path):
    with open(output_path, 'w+') as f:
        for (value, l) in lists:
            f.write('\t'.join([str(value), str(l), '#\n']))

    x = [value for (value, l) in lists]
    y = [l for (value, l) in lists]

    plt.scatter(x, y, alpha=0.5)
    plt.savefig(output_path.split('.')[0] +'_graph.png')
    plt.close()


def micro_lists_path_based_pedersen_similarity():
    similarities_function_names = ['path', 'lch', 'wup', 'res', 'jcn', 'lin']
    spearman = {}

    evaluator = SimilarityEvaluator('cosine_similarity')
    pretrained_embeddinds_path = 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(pretrained_embeddinds_path, binary=True)

    K = 15
    N_TEST = 2000
    TEST_SIZE = 7
    for measure in similarities_function_names:
        n_couple_clusters = retrieve_couples_divided_by_value_of_similarity(
            positive_input_path='data/similarity_pedersen_test/sister_terms/in_voc_sister_terms_positive.txt',
            negative_input_path='data/similarity_pedersen_test/sister_terms/in_voc_sister_terms_negative.txt',
            measure_name=measure)

        lenghts_sublists = [(value, len(n_couple_clusters[value])) for value in n_couple_clusters]
        save_clusters(lenghts_sublists,
                      output_path='data/similarity_pedersen_test/in_vocabulary_similarities/clusters_n_' + measure + '.txt')
        """min_len = min([len for (value, len) in lenghts_sublists])
        max_len = max([len for (value, len) in lenghts_sublists])
        avg = mean([len for (value, len) in lenghts_sublists])
        print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
        print('\n')"""


        k_clusters = Cluster.k_clusters_of_min_diameter(k=K, n_clusters=n_couple_clusters)
        lenghts_sublists = [(value, len(k_clusters[value])) for value in k_clusters]
        save_clusters(lenghts_sublists,
                      output_path='data/similarity_pedersen_test/in_vocabulary_similarities/clusters_k_' + measure + '.txt')
        """min_len = min([len for (value, len) in lenghts_sublists])
        max_len = max([len for (value, len) in lenghts_sublists])
        avg = mean([len for (value, len) in lenghts_sublists])
        print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
        print('----------------')"""

        ouput_path = 'data/similarity_pedersen_test/in_vocabulary_similarities/micro_lists/'+measure+'_micro_lists_test.txt'
        tests = collect_test_of_size(n_test=N_TEST, test_size=TEST_SIZE, k_clusters=k_clusters,
                                     ouput_path=ouput_path)
        """print('----------------')
        lenghts_sublists = [len(test) for test in tests]
        min_len = min(lenghts_sublists)
        max_len = max(lenghts_sublists)
        avg = mean(lenghts_sublists)
        print(lenghts_sublists)
        print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
        print('----------------')"""
        spearman[measure] = []
        for i in range(0, len(tests)):
            oracle = InVocCorrelationOracle(path=None)
            for d in tests[i]:
                (similarity_value, synset_couple) = d
                oracle.add_correlations(value=similarity_value, first=synset_couple.w1, second=synset_couple.w2)
            tester = PetersenInVocTester(oracle=oracle)

            spearman[measure].append(tester.spearman_correlation_model_predictions_and_oracle(
                model, evaluator,
                save_on_file=True,
                path='data/similarity_pedersen_test/in_vocabulary_similarities/micro_lists/'+measure+'_output_micro_lists_test.txt',
                mode='a+')
            )
        mu, std = gauss_distribution_of(
            [-x.correlation for x in spearman[measure]],
            measure,
            output_path='data/similarity_pedersen_test/in_vocabulary_similarities/micro_lists/'+measure+'_gauss_test.png')
        print(measure, mu, std)


"""
All SYNSET given a certain lemma as positive examples:
SEED 19
path 0.18010714285714288 0.40374546026385677
lch 0.26625000000000004 0.389187364706177
wup 0.3831071428571429 0.35195603382368373
res 0.5518035714285715 0.29875496270747864
jcn 0.3203750000000001 0.38588559315897386
lin 0.4305892857142858 0.345180571921629

SEED 99
path 0.16176785714285716 0.4091509528672063
lch 0.24971428571428578 0.39279209851149155
wup 0.3660714285714286 0.36836736460512154
res 0.5524285714285716 0.3041506574856935
jcn 0.3429642857142858 0.36789596015721626
lin 0.39348214285714295 0.3498472584899862

SEED 200
path 0.16967857142857148 0.39429282958124373
lch 0.2523928571428572 0.385635757548493
wup 0.37200000000000005 0.3592259677673197
res 0.5410357142857144 0.3012025197573276
jcn 0.32142857142857145 0.3735004029722166
lin 0.38446428571428576 0.367199090676812

SEED 1999
path 0.1883571428571429 0.39762997489974966
lch 0.26994642857142853 0.37233333785338224
wup 0.36398214285714287 0.36629666894625523
res 0.534625 0.30800661301458176
jcn 0.31857142857142867 0.3882762140118127
lin 0.42503571428571435 0.3586631908054903

SEED 5348
path 0.1689464285714286 0.4090850757127647
lch 0.26146428571428576 0.39276128538326993
wup 0.3827321428571429 0.3590416093325994
res 0.5368214285714287 0.303506377484015
la len del test finora + quella dei centri ancora disponibili e' minore della size del test
6 vs 7
jcn 0.31320438623210484 0.37332597088335856
lin 0.41498214285714297 0.3509075692117542
"""

"""
Only FIRST Synset given a certain lemma: if both noun and verb synsets are present, they are both taken
SEED 19
path 0.20155357142857147 0.3932326253508299
lch 0.31596428571428575 0.36674518311320015
wup 0.43269642857142865 0.33991888241152185
res 0.5838928571428573 0.2789928108758507
la len del test finora + quella dei centri ancora disponibili e' minore della size del test
6 vs 7
jcn 0.35266645996434387 0.37307132141642135
lin 0.4641250000000001 0.3288071354672579

SEED 99
path 0.1894107142857143 0.3955288821564084
lch 0.3189821428571429 0.3747325708837203
wup 0.4173750000000001 0.3485341346962582
res 0.5712321428571429 0.28897573725477166
la len del test finora + quella dei centri ancora disponibili e' minore della size del test
6 vs 7
jcn 0.3547619047619048 0.3654897624283194
lin 0.4528928571428572 0.3405994097296167

SEED 200
path 0.1769821428571429 0.39868329163392496
lch 0.30985714285714294 0.3774512132280767
wup 0.4326428571428572 0.34546073198981503
res 0.5886964285714287 0.27559705073712243
la len del test finora + quella dei centri ancora disponibili e' minore della size del test
6 vs 7
jcn 0.42530686863410816 0.3544045901944931
lin 0.4350892857142858 0.3480122009758584

SEED 1999
path 0.20307142857142862 0.3877318332169748
lch 0.33937500000000004 0.3641929044063022
wup 0.4374464285714287 0.3370352956256199
res 0.5715357142857144 0.29034855680505034
la len del test finora + quella dei centri ancora disponibili e' minore della size del test
6 vs 7
jcn 0.3867830589484865 0.35897016004513116
lin 0.45512500000000006 0.3319002719152984

SEED 5348
path 0.1878571428571429 0.39911796885285
lch 0.3067678571428572 0.3708666590066505
wup 0.4279642857142858 0.3432562985286532
res 0.5679642857142859 0.2931914272454161
la len del test finora + quella dei centri ancora disponibili e' minore della size del test
6 vs 7
jcn 0.4063317634746206 0.34119645116506636
lin 0.4582142857142858 0.34304514934596575
"""


class LastraDiazReader(LineReader):
    def readline(self, line):
        try:
            value = float(line[2])
            first = line[0]
            second = line[1]

            return value, first, second
        except ValueError:
            raise UnexpectedValueInLine(line)


class LastraDiazOracle(Oracle):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def collect_correlations(self, reader: LastraDiazReader, index_range=range(0, 3)):
        parser = Parser(self.path, ';')
        with parser:
            while True:
                line = parser.get_example_from_line_next_line(index_range)
                if not line:
                    break

                try:
                    value, first, second = reader.readline(line)
                    super().add_correlations(value, first, second)
                except UnexpectedValueInLine:
                    continue


class LastraDiazTester(Tester):
    def __init__(self, oracle: LastraDiazOracle):
        self.oracle = oracle

    def test_similarity_of_predictions(self, model: KeyedVectors, evaluator: SimilarityEvaluator, save_on_file=True,
                                       path='lastra_diaz_correlations.txt', mode=None):
        similarities = {}
        deletable = {}
        if save_on_file:
            header = '\t'.join(['id', 'first', 'second', 'oracle_value', 'model_value', '#\n'])
            writer = TestWriter(path, header, mode)

        for i in self.oracle.correlations:
            correlation = self.oracle.correlations[i]
            try:
                prediction_1 = model.word_vec(correlation['first'])
                prediction_2 = model.word_vec(correlation['second'])
            except KeyError:
                deletable[i] = correlation
                continue

            similarities[i] = evaluator.similarity_function(prediction_1, prediction_2).numpy()
            if save_on_file:
                writer.write_line(i, [correlation['first'], correlation['second']],
                                  [str(correlation['value']), str(similarities[i])])
        if save_on_file:
            writer.release()

        for i in deletable:
            del self.oracle.correlations[i]
        return similarities

    def spearman_correlation_model_predictions_and_oracle(self, model: KeyedVectors, evaluator: SimilarityEvaluator,
                                                          save_on_file=True,
                                                          path='diaz_correlations.txt',
                                                          mode=None):
        similarities = self.test_similarity_of_predictions(model, evaluator, save_on_file, path, mode)
        return spearmanr([similarities[key] for key in similarities], [self.oracle.correlations[key]['value']
                                                                       for key in self.oracle.correlations])


def diaz_tests(test_type, datasets_path, save_of_file=False):
    evaluator = SimilarityEvaluator('cosine_similarity')
    pretrained_embeddinds_path = 'data/pretrained_embeddings/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(pretrained_embeddinds_path, binary=True)

    results = {}
    for dataset in os.listdir(path=datasets_path):
        dataset_name = Dataset[dataset.split('_')[0]].value

        oracle = LastraDiazOracle(path=os.path.join(datasets_path, dataset))
        oracle.collect_correlations(LastraDiazReader())

        distinct_values = {}
        for i in oracle.correlations:
            correlation = oracle.correlations[i]
            distinct_values[correlation['value']] = 1

        len_test = len(oracle.correlations.keys())
        n_distinct_values = len(distinct_values)

        tester = LastraDiazTester(oracle)

        if save_of_file:
            path = test_type+'_'+dataset_name+'.txt'
        else:
            path = 'diaz_correlations.txt'

        spearman = tester.spearman_correlation_model_predictions_and_oracle(
            model, evaluator,
            save_on_file=save_of_file,
            path=path)

        # print(str(type(sequential).__name__) + ' --> ' + str(spearman_sequential))
        """print('--------------')
        print(dataset_name, str(type(model).__name__) + ' --> ' + str(spearman))
        print('--------------')"""

        results[dataset_name] = (- spearman.correlation, len_test, n_distinct_values)

    return results


def noun_verb_sim_relatedness_diaz_dataset(root):
    header = '\t'.join(['DATASET_NAME', 'SPEARMAN_CORR', 'LEN_TEST', 'DISTINCT_VALUES', '#\n'])
    output_path = 'diaz_tests_recap.txt'
    for test_type in os.listdir(root):
        results = diaz_tests(test_type,
                             datasets_path=os.path.join(root, test_type),
                             save_of_file=True)

        with open(output_path, mode='a+') as file:
            file.write(test_type.upper()+'\t#\n')
            file.write(header)
            for dataset_name in results:
                spearman_correlation, len_test, n_distinct_values = results[dataset_name]
                file.write('\t'.join([dataset_name, str(spearman_correlation),
                                      str(len_test), str(n_distinct_values), '#\n']))


noun_verb_sim_relatedness_diaz_dataset('data/similarity_lastra_diaz_test')
