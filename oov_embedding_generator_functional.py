import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import History
from baseline.BaselineAdditiveModel import BaselineAdditiveModel
from writer.writer_utility import ExampleWriter, POSAwareExampleWriter
from example_to_numpy.example_to_numpy import ExampleToNumpy, POSAwareExampleToNumpy
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

    dataset = list(zip(dataset_data, dataset_target, dataset_pos))
    random.shuffle(dataset)
    dataset_data, dataset_target, dataset_pos = zip(*dataset)
    return dataset_data, dataset_target, dataset_pos


def split_in(split_test: float, dataset_data, dataset_target, dataset_pos):
    TEST_SIZE = int(len(dataset_data) * split_test)

    test_data = np.array(dataset_data[0: TEST_SIZE])
    test_target = np.array(dataset_target[0: TEST_SIZE])
    test_pos = np.array(dataset_pos[0: TEST_SIZE])

    train_data = np.array(dataset_data[TEST_SIZE:])
    train_target = np.array(dataset_target[TEST_SIZE:])
    train_pos = np.array(dataset_pos[TEST_SIZE:])

    return (test_data, test_target, test_pos), (train_data, train_target, train_pos)


"""def compare_with_baseline(model_mse, baseline_type, test_data, test_target):
    if baseline_type == 'additive':
        baseline = BaselineAdditiveModel()
        for i in range(0, len(test_data)):
            baseline.process_example(target=test_target[i], data=test_data[i])

        return 1 - model_mse / baseline.calculate_mse()"""


def save(model, test, training):
    model.save('oov_sequential_predictor.h5')

    (test_data, test_target, test_pos) = test
    test_saver = POSAwareExampleToNumpy(data=test_data, target=test_target, pos=test_pos)
    test_saver.save_numpy_examples('data/test_oov_sequential_predictor.npz')

    (train_data, train_target, train_pos) = training
    train_saver = POSAwareExampleToNumpy(data=train_data, target=train_target, pos=train_pos)
    train_saver.save_numpy_examples('data/train_oov_sequential_predictor.npz')


def all_descendant_files_of(base):
    input_paths = []
    for root, dirs, files in os.walk(base, topdown=False):
        input_paths.extend([os.path.join(root, x) for x in files])
    return input_paths


def plot(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


base = "data/wordnet_definition/"
input_paths = all_descendant_files_of(base)

path = 'data/google_w2v_example.npz'
write_w2v_exaples_from_to(input_paths, path)

dataset_data, dataset_target, dataset_pos = load_dataset_from(path=path)
(test_data, test_target, test_pos), (train_data, train_target, train_pos) = split_in(0.10, dataset_data, dataset_target, dataset_pos)

FEATURES = 300
N_POS_TAG = 3

#pos_embedding = tf.keras.layers.Embedding(input_dim=N_POS_TAG, output_dim=10, sequence_lenght=1)

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


pos_one_hot = tf.keras.Input(shape=(N_POS_TAG,))
x = tf.keras.layers.Dense(10)(pos_one_hot)
#x3 = pos_embedding(pos_one_hot)
#x3 = tf.keras.layers.Flatten()(x3)
x3 = tf.keras.layers.Dense(300)(x)
x3 = tf.keras.layers.Dense(500, activation='relu')(x3)
x3 = tf.keras.layers.Dropout(rate=0.15)(x3)
x3 = tf.keras.layers.Dense(400, activation='tanh')(x3)

x = tf.keras.layers.Add()([x1, x2, x3])
output = tf.keras.layers.Dense(FEATURES)(x)

model = tf.keras.Model(inputs=[first_embedding, second_embedding, pos_one_hot], outputs=output)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.cosine_similarity,
    metrics=[tf.keras.metrics.mse, tf.keras.losses.cosine_similarity]
)

N_EPOCHS = 30
BATCH_SIZE = 32

history: History = model.fit(x=[train_data[:, 0], train_data[:, 1], train_pos], y=train_target, epochs=N_EPOCHS,
                             batch_size=BATCH_SIZE)
plot(history)

test_history = model.evaluate(x=[test_data[:, 0], test_data[:, 1], test_pos], y=test_target)


"""r = compare_with_baseline(test_history[1], 'additive', test_data, test_target)
print(f'R model against additive model:{r}')"""

model.save('oov_functional_predictor.h5')
