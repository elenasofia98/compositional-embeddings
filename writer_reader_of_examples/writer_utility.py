"""
Parser class creates objects able of:
- getting list of examples from a line-based file
- getting an example from next line in file
- getting an example from a given line
"""
from example_to_numpy.example_to_numpy import ExampleToNumpy, POSAwareExampleToNumpy
from preprocessing.w2v_preprocessing_embedding import PreprocessingWord2VecEmbedding, POSAwarePreprocessingWord2VecEmbedding, OOVExampleException
import os


class Parser:
    def __init__(self, file_path: str, word_separator: str):
        self.file_path = file_path
        self.word_separator = word_separator

    def __enter__(self):
        self.file = open(file=self.file_path)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def get_example_from_line_next_line(self, index_range):
        next_line = self.file.readline()
        if not next_line:
            return []
        return self.get_example_from_line(next_line, index_range)

    def get_example_from_line(self, line, index_range):
        splitted = line.split(self.word_separator)
        print(splitted)
        return [splitted[i] for i in index_range]


class ExampleWriter:
    def __init__(self, example_paths, separator, output_path, preprocessor: PreprocessingWord2VecEmbedding):
        self.example_paths = example_paths
        self.separator = separator
        self.output_path = output_path
        self.preprocessor = preprocessor

    def write_w2v_examples(self):
        tot_example = {}
        tot_error = {}

        saver = ExampleToNumpy()
        for path in self.example_paths:
            examples = 0
            errors = 0

            index_range = range(1, 4)
            parser = Parser(path, self.separator)
            with parser:
                while True:
                    words = parser.get_example_from_line_next_line(index_range)
                    if not words:
                        break
                    examples += 1
                    try:
                        example = self.preprocessor.get_vector_example(words)
                        saver.add_example(example)
                    except OOVExampleException:
                        errors += 1

            tot_example[path] = examples
            tot_error[path] = errors

        print(tot_example)
        print(tot_error)

        saver.save_numpy_examples(self.output_path)


class POSAwareExampleWriter(ExampleWriter):
    def __init__(self, example_paths, separator, output_path, preprocessor: POSAwarePreprocessingWord2VecEmbedding):
        super().__init__(example_paths, separator, output_path, preprocessor)
        self.pos_tags_example = self.get_pos_tags_from_paths()

    def write_w2v_examples(self):
        tot_example = {}
        tot_error = {}

        saver = POSAwareExampleToNumpy()
        for path in self.example_paths:
            examples = 0
            errors = 0

            index_range = range(1, 4)
            parser = Parser(path, self.separator)
            with parser:
                while True:
                    words = parser.get_example_from_line_next_line(index_range)
                    if not words:
                        break
                    examples += 1
                    try:
                        example = self.preprocessor.get_vector_example(words, self.pos_tags_example[path])
                        saver.add_example(example)
                    except OOVExampleException:
                        errors += 1

            tot_example[path] = examples
            tot_error[path] = errors

        print(tot_example)
        print(tot_error)

        saver.save_numpy_examples(self.output_path)

    def get_pos_tags_from_paths(self):
        pos_tags_by_path = {}
        for path in self.example_paths:
            pos_tags_by_path[path]= self.get_pos_tags_from_path(path)
        return pos_tags_by_path


    def get_pos_tags_from_path(self, path):
        split = os.path.normpath(path).split(os.path.sep)
        target_pos = split[-2]
        basename_split = split[-1].split('_')
        return {'target_pos': target_pos, 'w1_pos': basename_split[0], 'w2_pos': basename_split[1]}
