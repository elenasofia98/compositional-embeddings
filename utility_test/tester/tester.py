from utility_test.similarity_evaluator.similarity_evaluator import SimilarityEvaluator


class Tester:
    def collect_similarity_of_predictions(self, model, evaluator: SimilarityEvaluator, save_on_file=True,
                                          path='correlations.txt', mode=None):
        pass

    def spearman_correlation_model_predictions_and_oracle(self, model, evaluator: SimilarityEvaluator,
                                                          save_on_file=True):
        pass


class TestWriter:
    def __init__(self, path, header, mode=None):
        self.separator = '\t'
        if mode is None:
            self.mode = 'w+'
        else:
            self.mode = mode
        self.create_file(path)
        self.write_header(header)

    def write_header(self, header):
        self.file.write(header)

    def create_file(self, path):
        self.file = open(path, mode=self.mode)

    def write_line(self, index, oracle_line, correlations):
        oracle_line = self.separator.join([str(x) for x in oracle_line])
        correlations = self.separator.join([str(x) for x in correlations])

        self.file.write(self.separator.join([str(index), oracle_line, correlations, '#\n']))

    def write_lines(self, lines):
        self.file.writelines(lines)

    def release(self):
        self.file.close()


class LineReader:
    def readline(self, line):
        pass


class UnexpectedValueInLine(ValueError):
    def __init__(self, line):
        message = 'Value Error occurred during the reading of line:\n' + str(line)
        super().__init__(message)