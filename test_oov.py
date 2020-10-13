import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors

from preprocessing.w2v_preprocessing_embedding import PreprocessingWord2VecEmbedding, OOVWordException
from writer.writer_utility import Parser

class BadDataException(Exception):
    def __init__(self):
        message = 'Bad structure for given test: each input data must contain 2 word'
        super().__init__(message)


class TestToNumpy:
    def __init__(self, data=None, goal=None):
        if data is None and goal is None:
            self.data = []
            self.goal = []
        else:
            self.data = data
            self.goal = goal

    def add_data(self, data, goal):
        if len(data) != 2:
            raise BadDataException()

        self.data.append(np.array(data))
        self.goal.append(goal)

    def save_numpy_examples(self, path):
        np.savez(path, data=self.data, goal=self.goal)


class TestWriter:
    def __init__(self, example_paths, separator, output_path, preprocessor: PreprocessingWord2VecEmbedding):
        self.example_paths = example_paths
        self.separator = separator
        self.output_path = output_path
        self.preprocessor = preprocessor

    def write_w2v_examples(self):
        tot_data = {}
        tot_error = {}

        saver = TestToNumpy()
        for path in self.example_paths:
            data = 0
            errors = 0

            index_range = range(1, 4)
            parser = Parser(path, self.separator)
            with parser:
                while True:
                    words = parser.get_example_from_line_next_line(index_range)
                    print(words)
                    if not words:
                        break
                    data += 1
                    try:
                        example = self.preprocessor.get_vectors(words[1:])
                        saver.add_data(example, words[0])
                    except OOVWordException:
                        errors += 1

            tot_data[path] = data
            tot_error[path] = errors

        print(tot_data)
        print(tot_error)

        saver.save_numpy_examples(self.output_path)


def write_w2v_test_from_to(paths, output_path):
    writer = TestWriter(example_paths=paths, separator='\t', output_path=output_path,
                           preprocessor=PreprocessingWord2VecEmbedding(
                               "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin",
                               binary=True)
                           )
    writer.write_w2v_examples()

def load_test_from(path):
    with np.load(path, allow_pickle=True) as data:
        dataset_data = data['data']
        goal_data = data['goal']
    return dataset_data, goal_data


path = 'data/oov_google_w2v_example.npz'
write_w2v_test_from_to(['data/oov_definition.txt'], path)

dataset_data, goal_data = load_test_from(path=path)

print(goal_data.shape)
print(dataset_data.shape)
print(dataset_data[0].shape)


model: tf.keras.models.Sequential = tf.keras.models.load_model('oov_sequential_predictor.h5')
predictions = model.predict(dataset_data)

w2v_model = KeyedVectors.load_word2vec_format("data/pretrained_embeddings/GoogleNews-vectors-negative300.bin", binary=True)
w2v_model.init_sims(replace=True)

for i in range(len(predictions)):
    closer_to_prediction = w2v_model.most_similar(positive=[predictions[i]])
    print(closer_to_prediction)
    print(f'goal={goal_data[i]}')
    print('---------------------------------------')