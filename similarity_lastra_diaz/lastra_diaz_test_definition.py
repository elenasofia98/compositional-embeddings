import os
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from utility_test.oracle.oracle import Oracle
from utility_test.tester.tester import Tester, TestWriter, LineReader, UnexpectedValueInLine
from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator
from writer_reader_of_examples.writer_utility import Parser
from scipy.stats import spearmanr
from similarity_lastra_diaz.collect_lastra_diaz_similarities import Dataset


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

            similarities[i] = evaluator.similarity_function(prediction_1, prediction_2)
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


def diaz_tests(model: KeyedVectors, test_type, datasets_path, output_path_root, save_of_file=False):
    evaluator = SimilarityEvaluator('cosine_similarity')

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
            path = os.path.join(output_path_root, test_type+'_'+dataset_name+'.txt')
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


def noun_verb_sim_relatedness_diaz_dataset(model: KeyedVectors, root, output_path_root):
    header = '\t'.join(['DATASET_NAME', 'SPEARMAN_CORR', 'LEN_TEST', 'DISTINCT_VALUES', '#\n'])

    tests = os.listdir(root)
    tests.remove('results')

    for test_type in tests:
        results = diaz_tests(model,
                             test_type,
                             datasets_path=os.path.join(root, test_type),
                             output_path_root=output_path_root,
                             save_of_file=True)

        with open(os.path.join(output_path_root, test_type), mode='a+') as file:
            file.write(test_type.upper()+'\t#\n')
            file.write(header)
            for dataset_name in results:
                spearman_correlation, len_test, n_distinct_values = results[dataset_name]
                file.write('\t'.join([dataset_name, str(spearman_correlation),
                                      str(len_test), str(n_distinct_values), '#\n']))
