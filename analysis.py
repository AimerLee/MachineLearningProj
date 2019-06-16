import pickle

import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix


def plot_validation_graph():
    data = np.load('./result/validate_acc.npy')
    mean_acc = data.mean(axis=0)
    best_n = 1 + 2 * mean_acc.argmax()
    x = np.arange(1, 100, 2)
    plt.plot(x, data[0, :], label='Validation 1')
    plt.plot(x, data[1, :], label='Validation 2')
    plt.plot(x, data.mean(axis=0), label='Mean Accuracy')
    plt.xlim(0, 100)
    plt.ylim(0, 0.6)
    plt.title("Cross Validation of KNN")
    plt.xlabel("Value of N")
    plt.ylabel("Accuracy")
    plt.vlines(best_n, 0, mean_acc.max(), linestyles='dotted', label='n = {:d}'.format(best_n))
    plt.legend()
    plt.show()


def plot_cm():
    tags = ['Hate', 'Fear', 'Sad', 'Neutral', 'Happy']

    def map_labels(value):
        return tags[int(value)]

    # KNN
    with open('./result/knn_result.npy', 'rb') as f:
        result_dict = pickle.load(f)
        p_labels = list(map(map_labels, result_dict['p_labels']))
        true_labels = list(map(map_labels, result_dict['test_labels']))

        plot_confusion_matrix(true_labels, p_labels, normalize=True, labels=tags)
        plt.show()

    # SVM
    result_dict = np.load('./result/svm_result.npy', allow_pickle=True).tolist()
    p_labels = list(map(map_labels, result_dict['p_labels']))
    true_labels = list(map(map_labels, result_dict['test_labels']))
    plot_confusion_matrix(true_labels, p_labels, normalize=True, labels=tags)
    plt.show()

    # MLP
    true_labels = list(map(map_labels, np.load('./result/MLP_test.npy')))
    p_labels = list(map(map_labels, np.load('./result/MLP_pred.npy')))
    plot_confusion_matrix(true_labels, p_labels, normalize=True, labels=tags)
    plt.show()

    # LR
    true_labels = list(map(map_labels, np.load('./result/lr_test.npy')))
    p_labels = list(map(map_labels, np.load('./result/lr_pred.npy')))
    plot_confusion_matrix(true_labels, p_labels, normalize=True, labels=tags)
    plt.show()


plot_validation_graph()
plot_cm()
