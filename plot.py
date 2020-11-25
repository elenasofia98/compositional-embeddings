import matplotlib.pyplot as plt
import numpy
import csv

def plot_scattered(x, ys):
    plt.legend
    ax = plt.gca()
    ax.scatter(x, ys[0], color="b")
    ax.scatter(x, ys[1], color="r")
    plt.show()


x = []
ys = [[], []]
with open('data/pedersen_test/results_spearman_test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x.append(int(row[0]))
        ys[0].append(-float(row[1]))
        ys[1].append(-float(row[2]))

plot_scattered(x, ys)