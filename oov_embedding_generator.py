import tensorflow as tf
import numpy as np
import random
import os

from tensorflow.python.keras.callbacks import History
from gensim.models import KeyedVectors

from baseline.BaselineAdditiveModel import BaselineAdditiveModel
from example_to_numpy.example_to_numpy import ExampleToNumpy
from preprocessing.w2v_preprocessing_embedding import PreprocessingWord2VecEmbedding
from writer.writer_utility import ExampleWriter


def write_w2v_exaples_from_to(paths, output_path, pretrained_model_path):
    # pretrained_layer_wiki = PreprocessingWord2VecEmbedding("data/enwiki_20180420_300d.txt", binary=False)
    writer = ExampleWriter(example_paths=paths, separator='\t', output_path=output_path,
                           preprocessor=PreprocessingWord2VecEmbedding(
                               pretrained_model_path,
                               binary=True)
                           )
    writer.write_w2v_examples()


def load_dataset_from(path):
    with np.load(path, allow_pickle=True) as data:
        dataset_data = data['data']
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


def compare_with_baseline(model_mse, baseline_type, test_data, test_target):
    if baseline_type == 'additive':
        baseline = BaselineAdditiveModel()
        for i in range(0, len(test_data)):
            baseline.process_example(target=test_target[i], data=test_data[i])

        return 1 - model_mse / baseline.calculate_mse()


def save(model_path, model, test, training):
    model.save(model_path)

    (test_data, test_target) = test
    test_saver = ExampleToNumpy(data=test_data, target=test_target)
    test_saver.save_numpy_examples('data/test_oov_sequential_predictor.npz')

    (train_data, train_target) = training
    train_saver = ExampleToNumpy(data=train_data, target=train_target)
    train_saver.save_numpy_examples('data/train_oov_sequential_predictor.npz')


base = "data/wordnet_definition/n/"
input_paths = [base + x for x in os.listdir(base)]
path = 'data/google_w2v_example.npz'
write_w2v_exaples_from_to(input_paths, path, "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin")


dataset_data, dataset_target = load_dataset_from(path=path)
(test_data, test_target), (train_data, train_target) = split_in(0.10, dataset_data, dataset_target)

print(train_data.shape)
print(train_data[0].shape)
print(f'v{train_data[0][1]}')
print(np.linalg.norm(train_data[0][1]))

print(train_target[0].shape)


WORDS, FEATURES = 2, 300

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(WORDS, FEATURES)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(600, activation='relu'),
    tf.keras.layers.Dropout(rate=0.15),
    tf.keras.layers.Dense(900, activation='relu'),
    tf.keras.layers.Dropout(rate=0.15),
    tf.keras.layers.Dense(1200, activation='relu'),
    tf.keras.layers.Dropout(rate=0.15),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(rate=0.15),
    tf.keras.layers.Dense(FEATURES)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.cosine_similarity,
    metrics=[tf.keras.metrics.mse, tf.keras.losses.cosine_similarity]
)

N_EPOCHS = 25
BATCH_SIZE = 64

history: History = model.fit(x=train_data, y=train_target, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

test_history = model.evaluate(x=test_data, y=test_target)

r = compare_with_baseline(test_history[1], 'additive', test_data, test_target)
print(f'R, current model against additive model:{r}')


save('oov_sequential_predictor_noun_only.h5', model, (test_data, test_target), (train_data, train_target))
