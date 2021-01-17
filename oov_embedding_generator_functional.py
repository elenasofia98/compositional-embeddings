import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from tensorflow.python.keras.callbacks import History

from similarity_pedersen.pedersen_similarities import ReaderSynsetCouples, SynsetOOVCouple, ReaderSynsetOOVCouple
from writer_reader_of_examples.writer_utility import POSAwareExampleWriter
from example_to_numpy.example_to_numpy import ExampleToNumpy, POSAwareExampleToNumpy
from preprocessing.w2v_preprocessing_embedding import POSAwarePreprocessingWord2VecEmbedding


def write_w2v_example_excluding(test_path, input_paths, output_path):
    test_couples = ReaderSynsetOOVCouple.read(test_path)
    test_dict = {'_'.join(couple.first):1 for couple in test_couples}

    paths = []
    for path in input_paths:
        new_input_path = path.split('.')[0] + '_test_excluded.txt'
        paths.append(new_input_path)

        if not os.path.exists(new_input_path):
            with open(path, 'r+') as input:
                with open(new_input_path, 'w+') as new_input:
                    output_lines = []
                    for line in input.readlines():
                        split = line.split('\t')
                        if '_'.join(split[2:4]) not in test_dict:
                            output_lines.append(line)

                    new_input.writelines(output_lines)

    #write_w2v_exaples_from_to(paths, output_path)


def write_w2v_exaples_from_to(paths, output_path):
    writer = POSAwareExampleWriter(example_paths=paths, separator='\t', output_path=output_path,
                                   preprocessor=POSAwarePreprocessingWord2VecEmbedding(
                                       "data/pretrained_embeddings/GoogleNews-vectors-negative300.bin",
                                       binary=True)
                                   )
    writer.write_w2v_examples()


def write_w2v_example_excluding(test_paths, input_paths, output_path):
    test_couples = []
    for test_path in test_paths:
        test_couples.extend(ReaderSynsetOOVCouple.read(test_path))

    test_dict = {'_'.join(couple.first): 1 for couple in test_couples}

    paths = []
    for path in input_paths:
        new_input_path = path.split('.')[0] + '_no_sister_terms_test.txt'
        paths.append(new_input_path)
        with open(path, 'r+') as input:
            with open(new_input_path, 'w+') as new_input:
                output_lines = []
                for line in input.readlines():
                    split = line.split('\t')
                    if '_'.join(split[2:4]) not in test_dict:
                        output_lines.append(line)

                new_input.writelines(output_lines)

    write_w2v_exaples_from_to(paths, output_path)


def load_dataset_from(path):
    with np.load(path, allow_pickle=True) as data:
        dataset_data = data['data']
        dataset_target = data['target']
        dataset_target_pos = data['target_pos']
        dataset_w1_pos = data['w1_pos']
        dataset_w2_pos = data['w2_pos']

    dataset = list(zip(dataset_data, dataset_target, dataset_target_pos, dataset_w1_pos, dataset_w2_pos))
    random.shuffle(dataset)
    dataset_data, dataset_target, dataset_target_pos, dataset_w1_pos, dataset_w2_pos = zip(*dataset)
    return dataset_data, dataset_target, dataset_target_pos, dataset_w1_pos, dataset_w2_pos


def split_in(split_test: float, dataset_data, dataset_target, dataset_target_pos, dataset_w1_pos, dataset_w2_pos):
    TEST_SIZE = int(len(dataset_data) * split_test)

    test_data = np.array(dataset_data[0: TEST_SIZE])
    test_target = np.array(dataset_target[0: TEST_SIZE])
    test_target_pos = np.array(dataset_target_pos[0: TEST_SIZE])
    test_w1_pos = np.array(dataset_w1_pos[0: TEST_SIZE])
    test_w2_pos = np.array(dataset_w2_pos[0: TEST_SIZE])

    train_data = np.array(dataset_data[TEST_SIZE:])
    train_target = np.array(dataset_target[TEST_SIZE:])
    train_target_pos = np.array(dataset_target_pos[TEST_SIZE:])
    train_w1_pos = np.array(dataset_w1_pos[TEST_SIZE:])
    train_w2_pos = np.array(dataset_w2_pos[TEST_SIZE:])

    return (test_data, test_target, test_target_pos, train_w1_pos, train_w2_pos), \
           (train_data, train_target, train_target_pos, test_w1_pos, test_w2_pos)


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


test_paths = []
for seed in ['19', '99', '200', '1999', '5348']:
    for t in ['positive', 'negative']:
        test_paths.append('data/similarity_pedersen_test/oov_sister_terms_with_definitions/seed_'+seed+'/oov_definition_sister_terms_'+t+'.txt')

base = "data/wordnet_definition/"
input_paths = all_descendant_files_of(base)

"""dataset_path = 'data/google_w2v_example.npz'
write_w2v_exaples_from_to(input_paths, dataset_path)"""

dataset_path = 'data/google_w2v_example_no_sister_terms_test.npz'
write_w2v_example_excluding(test_paths=test_paths, input_paths=input_paths, output_path=dataset_path)

dataset_data, dataset_target, dataset_target_pos, dataset_w1_pos, dataset_w2_pos = load_dataset_from(path=dataset_path)
test, train = split_in(0.20, dataset_data, dataset_target, dataset_target_pos, dataset_w1_pos, dataset_w2_pos)
(test_data, test_target, test_target_pos, train_w1_pos, train_w2_pos) = test
(train_data, train_target, train_target_pos, test_w1_pos, test_w2_pos) = train

FEATURES = 300
N_POS_TAG = 3

# pos_embedding = tf.keras.layers.Embedding(input_dim=N_POS_TAG, output_dim=10, sequence_lenght=1)

first_embedding = tf.keras.layers.Input(shape=(FEATURES,))
x1 = tf.keras.layers.Dense(500)(first_embedding)
x1 = tf.keras.layers.LeakyReLU(alpha=0.5)(x1)
x1 = tf.keras.layers.Dropout(rate=0.15)(x1)
x1 = tf.keras.layers.Dense(300)(x1)

second_embedding = tf.keras.layers.Input(shape=(FEATURES,))
x2 = tf.keras.layers.Dense(500)(second_embedding)
x2 = tf.keras.layers.LeakyReLU(alpha=0.5)(x2)
x2 = tf.keras.layers.Dropout(rate=0.15)(x2)
x2 = tf.keras.layers.Dense(300)(x2)

x = tf.keras.layers.Add()([x1, x2])
x = tf.keras.layers.Dense(400)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.5)(x)
x = tf.keras.layers.Dropout(rate=0.15)(x)
x = tf.keras.layers.Dense(300)(x)

w1_pos_one_hot = tf.keras.Input(shape=(N_POS_TAG,))
w1_x2 = tf.keras.layers.Dense(100)(w1_pos_one_hot)

w2_pos_one_hot = tf.keras.Input(shape=(N_POS_TAG,))
w2_x2 = tf.keras.layers.Dense(100)(w2_pos_one_hot)

target_pos_one_hot = tf.keras.Input(shape=(N_POS_TAG,))
target_x2 = tf.keras.layers.Dense(100)(target_pos_one_hot)

t = tf.keras.layers.Concatenate()([w1_x2, w2_x2, target_x2])
t = tf.keras.layers.Dense(400)(t)
t = tf.keras.layers.LeakyReLU(alpha=0.5)(t)
t = tf.keras.layers.Dropout(rate=0.15)(t)
t = tf.keras.layers.Dense(300)(t)

x = tf.keras.layers.Concatenate()([x, t])
x = tf.keras.layers.Dense(1000)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.5)(x)
x = tf.keras.layers.Dropout(rate=0.15)(x)
x = tf.keras.layers.Dense(1200)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.5)(x)
x = tf.keras.layers.Dropout(rate=0.15)(x)
output = tf.keras.layers.Dense(300)(x)

model = tf.keras.Model(inputs=[first_embedding, w1_pos_one_hot, second_embedding, w2_pos_one_hot, target_pos_one_hot],
                       outputs=output)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.cosine_similarity,
    metrics=[tf.keras.metrics.mse, tf.keras.losses.cosine_similarity]
)

N_EPOCHS = 25
BATCH_SIZE = 64

history: History = model.fit(x=[train_data[:, 0], train_w1_pos, train_data[:, 1], train_w2_pos, train_target_pos],
                             y=train_target, epochs=N_EPOCHS,
                             batch_size=BATCH_SIZE)
plot(history)

test_history = model.evaluate(x=[test_data[:, 0], test_w1_pos, test_data[:, 1], test_w2_pos, test_target_pos],
                              y=test_target)

"""r = compare_with_baseline(test_history[1], 'additive', test_data, test_target)
print(f'R model against additive model:{r}')"""

model.save('oov_functional_predictor_no_sister_terms_test.h5')
