import random
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import nltk

from cluster.cluster import ClusterMinDiam
from similarity_pedersen.collect_pedersen_similarities import voc_sim, retrieve_in_voc_couples_divided_by_value_of_similarity
from similarity_pedersen.pedersen_similarities import InformationContent
from utility_test.distribution.distributions import Gauss
from utility_test.tester.tester import Tester, TestWriter, LineReader, UnexpectedValueInLine
from utility_test.oracle.oracle import Oracle
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from writer_reader_of_examples.writer_utility import Parser


def append_fixed_len(positives_input_path, negatives_input_path, output_path, expected_len):
    with open(positives_input_path, 'r') as f1:
        with open(negatives_input_path, 'r') as f2:
            with open(output_path, 'w+') as output:
                positives = f1.readlines()
                negatives = f2.readlines()

                positives_len = len(positives)
                negatives_len = len(negatives)
                for positive in positives:
                    if random.uniform(0, 1) <= expected_len / (2 * positives_len):
                        output.write(positive)
                for negative in negatives:
                    if random.uniform(0, 1) <= expected_len / (2 * negatives_len):
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

    def collect_similarity_of_predictions(self, model: KeyedVectors, evaluator: SimilarityEvaluator, save_on_file=True,
                                          path='petersen_correlations.txt', mode=None):
        similarities = {}

        if save_on_file:
            header = '\t'.join(['id', 'first', 'second', 'oracle_value', 'model_value', '#\n'])
            writer = TestWriter(path, header, mode)

        for i in self.oracle.correlations:
            correlation = self.oracle.correlations[i]

            prediction_1 = model.word_vec(correlation['first'])
            prediction_2 = model.word_vec(correlation['second'])

            similarities[i] = evaluator.similarity_function(prediction_1, prediction_2)

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
        similarities = self.collect_similarity_of_predictions(model, evaluator, save_on_file, path, mode)
        return spearmanr([similarities[key] for key in similarities], [self.oracle.correlations[key]['value']
                                                                       for key in self.oracle.correlations])


"""
Ogni volta si ricalcolano sia le coppie che il valore della loro similitudine: con il seed fissato per gli esempi postivi di dimilitudine non ha molto senso
Risultati inaffidabili a causa della ripetizione frequente di valori per la misura basaata sul dataset_path nella tassonomia di wordnet
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
            append_fixed_len(positives_input_path=positive_sim_examples_file,
                             negatives_input_path=negative_sim_examples_file,
                             output_path=output_path,
                             expected_len=2000)

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


def collect_test_of_size(n_test, test_size, k_clusters: dict, ouput_path=None):
    tests = []
    exists_available = True
    i = 0

    while i in range(0, n_test) and exists_available:
        test = []
        available_centers = [center for center in k_clusters if len(k_clusters[center]) != 0]

        for j in range(0, test_size):
            if j + len(available_centers) < test_size:
                """print('la len del test finora + quella dei centri ancora disponibili e\' minore della size del test')
                print(str(j + len(available_centers)) + ' vs ' + str(test_size))"""
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
                    output.write('\t'.join([str(i + 1), couple.s1.name(), couple.s2.name(),
                                            couple.w1, couple.w2, couple.s_pos, str(similarity_value), '#\n']))
    return tests


def save_clusters(lists, output_path):
    with open(output_path, 'w+') as f:
        for (value, l) in lists:
            f.write('\t'.join([str(value), str(l), '#\n']))

    x = [value for (value, l) in lists]
    y = [l for (value, l) in lists]

    plt.scatter(x, y, alpha=0.5)
    plt.savefig(output_path.split('.')[0] + '_graph.png')
    plt.close()


"""
    pretrained_embeddings_path = 'data/pretrained_embeddings/cc.en.300.bin'
    model = FastText.load_fasttext_format(pretrained_embeddings_path)
    model: FastTextKeyedVectors = model.wv

    base = 'data/similarity_pedersen_test/_fasttext'
    destination_dir = 'in_vocabulary_similarities_vocabulary_only/micro_lists/clustered_values'
"""


def micro_lists_in_voc_pedersen_similarity(model, root_data_model, destination_dir, similarities_function_names=None):
    if similarities_function_names is None:
        similarities_function_names = ['path', 'lch', 'wup', 'res', 'jcn', 'lin']
    spearman = {}

    evaluator = SimilarityEvaluator('cosine_similarity')

    root_data_model = root_data_model + '/'
    destination_dir = destination_dir + '/'

    K = 15
    N_TEST = 2000
    TEST_SIZE = 7

    for seed in ['19', '99', '200', '1999', '5348']:
        for measure in similarities_function_names:
            seed_dir = destination_dir + 'seed_' + seed + '/'
            if not os.path.exists(root_data_model + seed_dir):
                os.mkdir(root_data_model + seed_dir)

            n_couple_clusters = retrieve_in_voc_couples_divided_by_value_of_similarity(
                positive_input_path=root_data_model + 'sister_terms/seed_' + seed + '/in_voc_sister_terms_positive.txt',
                negative_input_path=root_data_model + 'sister_terms/seed_' + seed + '/in_voc_sister_terms_negative.txt',
                measure_name=measure)

            """lengths_sublists = [(value, len(n_couple_clusters[value])) for value in n_couple_clusters]
            save_clusters(lists=lengths_sublists,
                          output_path=root_data_model + 'in_vocabulary_similarities/seed_' + seed + '_clusters_n_' + measure + '.txt')
            """"""min_len = min([len for (value, len) in lengths_sublists])
            max_len = max([len for (value, len) in lengths_sublists])
            avg = mean([len for (value, len) in lengths_sublists])
            print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
            print('\n')"""

            k_clusters = ClusterMinDiam.k_clusters_of_min_diameter(k=K, n_clusters=n_couple_clusters)

            """lengths_sublists = [(value, len(k_clusters[value])) for value in k_clusters]
            save_clusters(lists=lengths_sublists,
                          output_path=root_data_model + 'in_vocabulary_similarities/seed_' + seed + '_clusters_k_' + measure + '.txt')
            """"""min_len = min([len for (value, len) in lengths_sublists])
            max_len = max([len for (value, len) in lengths_sublists])
            avg = mean([len for (value, len) in lengths_sublists])
            print(f'min_len={min_len}, max_len={max_len}, avg={avg}')
            print('----------------')"""

            output_path = root_data_model + seed_dir + measure + '_micro_lists_test.txt'
            tests = collect_test_of_size(n_test=N_TEST, test_size=TEST_SIZE, k_clusters=k_clusters,
                                         ouput_path=output_path)
            """print('----------------')
            lengths_sublists = [len(test) for test in tests]
            min_len = min(lengths_sublists)
            max_len = max(lengths_sublists)
            avg = mean(lengths_sublists)
            print(lengths_sublists)
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
                    path=root_data_model + seed_dir + measure + '_output_micro_lists_test.txt',
                    mode='a+')
                )

            distribution = Gauss(data=[-x.correlation for x in spearman[measure]])
            distribution.save(output_path=root_data_model + seed_dir + measure + '_gauss_test.png',
                              title=f"{measure} mini-lists spearman results")
            print('\t'.join([seed, measure, str(distribution.mu), str(distribution.std)]))


def ic_based_measure_on(information_content_root_path, model, root_data_model, destination_dir):
    for information_content_name in os.listdir(information_content_root_path):
        print(information_content_name)
        if information_content_name.split('.')[1] != 'dat':
            continue

        ic_dir = os.path.join(destination_dir, information_content_name.split('.')[0])
        if not os.path.exists(os.path.join(root_data_model, ic_dir)):
            os.mkdir(os.path.join(root_data_model, ic_dir))

        InformationContent.set_information_content(information_content_name)
        micro_lists_in_voc_pedersen_similarity(model, root_data_model, destination_dir,
                                               similarities_function_names=['res', 'jcn', 'lin'])