import tensorflow as tf
import numpy as np
import random
import os

from tensorflow.python.keras.callbacks import History
from baseline.BaselineAdditiveModel import BaselineAdditiveModel
from writer.writer_utility import ExampleWriter, POSAwareExampleWriter
from example_to_numpy.example_to_numpy import ExampleToNumpy
from preprocessing.w2v_preprocessing_embedding import POSAwarePreprocessingWord2VecEmbedding
from gensim.models import KeyedVectors


def write_w2v_exaples_from_to(paths, output_path):
    writer = POSAwareExampleWriter(example_paths=paths, separator='\t', output_path=output_path,
                                   preprocessor=POSAwarePreprocessingWord2VecEmbedding(
                                       "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin",
                                       binary=True)
                                   )
    writer.write_w2v_examples()


def load_dataset_from(path):
    with np.load(path, allow_pickle=True) as data:
        dataset_data = data['data']
        dataset_pos = data['pos']
        dataset_target = data['target']

    dataset = list(zip(dataset_data, dataset_target))
    random.shuffle(dataset)
    dataset_data, dataset_target = zip(*dataset)
    return dataset_data, dataset_target


def split_in(split_test: float, dataset_data, dataset_target):
    TEST_SIZE = int(len(dataset_data) * split_test)

    test_data = np.array(dataset_data[0: TEST_SIZE])
    test_target = np.array(dataset_target[0: TEST_SIZE])

    train_data = np.array(dataset_data[TEST_SIZE:])
    train_target = np.array(dataset_target[TEST_SIZE:])

    return (test_data, test_target), (train_data, train_target)


"""def compare_with_baseline(model_mse, baseline_type, test_data, test_target):
    if baseline_type == 'additive':
        baseline = BaselineAdditiveModel()
        for i in range(0, len(test_data)):
            baseline.process_example(target=test_target[i], data=test_data[i])

        return 1 - model_mse / baseline.calculate_mse()"""


def save(model, test, training):
    model.save('oov_sequential_predictor.h5')

    (test_data, test_target) = test
    test_saver = ExampleToNumpy(data=test_data, target=test_target)
    test_saver.save_numpy_examples('data/test_oov_sequential_predictor.npz')

    (train_data, train_target) = training
    train_saver = ExampleToNumpy(data=train_data, target=train_target)
    train_saver.save_numpy_examples('data/train_oov_sequential_predictor.npz')


def all_descendant_files_of(base):
    input_paths = []
    for root, dirs, files in os.walk(base, topdown=False):
        input_paths.extend([os.path.join(root, x) for x in files])
    return input_paths


base = "data/wordnet_definition/"
input_paths = all_descendant_files_of(base)

path = 'data/google_w2v_example.npz'
write_w2v_exaples_from_to(input_paths, path)

dataset_data, dataset_target = load_dataset_from(path=path)
(test_data, test_target), (train_data, train_target) = split_in(0.10, dataset_data, dataset_target)

FEATURES = 300

first_embedding = tf.keras.layers.Input(shape=(FEATURES,))
x1 = tf.keras.layers.Dense(300)(first_embedding)
x1 = tf.keras.layers.Dense(500, activation='relu')(x1)
x1 = tf.keras.layers.Dropout(rate=0.15)(x1)
x1 = tf.keras.layers.Dense(400, activation='tanh')(x1)

second_embedding = tf.keras.layers.Input(shape=(FEATURES,))
x2 = tf.keras.layers.Dense(300)(second_embedding)
x2 = tf.keras.layers.Dense(500, activation='relu')(x2)
x2 = tf.keras.layers.Dropout(rate=0.15)(x2)
x2 = tf.keras.layers.Dense(400, activation='tanh')(x2)

x = tf.keras.layers.Add()([x1, x2])
output = tf.keras.layers.Dense(FEATURES)(x)

model = tf.keras.Model(inputs=[first_embedding, second_embedding], outputs=output)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.cosine_similarity,
    metrics=[tf.keras.metrics.mse, tf.keras.losses.cosine_similarity]
)

N_EPOCHS = 30
BATCH_SIZE = 512

history: History = model.fit(x=[train_data[:, 0], train_data[:, 1]], y=train_target, epochs=N_EPOCHS,
                             batch_size=BATCH_SIZE)

test_history = model.evaluate(x=[test_data[:, 0], test_data[:, 1]], y=test_target)

"""r = compare_with_baseline(test_history[1], 'additive', test_data, test_target)
print(f'R model against additive model:{r}')"""

model.save('oov_functional_predictor.h5')
