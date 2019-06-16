import os
import numpy as np

from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# The ground truth of our dataset
def train_data():
    labels = ((4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0))

    for dir in range(1, 11):
        for i in range(3):
            for j in range(10):
                yield np.load(os.path.join('data', str(dir), str(i), 'DE_' + str(j) + '.npy')), labels[i][j]


def test_data():
    labels = ((4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0),
              (2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0))

    for dir in range(1, 11):
        for i in range(3):
            for j in range(10, 15):
                yield np.load(os.path.join('data', str(dir), str(i), 'DE_' + str(j) + '.npy')), labels[i][j]


def reshape(data):
    for i in range(data.shape[2]):
        yield data[:, :, i].reshape(310)


def load_data():
    x_train = []
    y_train = []

    for (data, l) in train_data():
        for d in reshape(data):
            x_train.append(d)
            y_train.append(l)

    x_train = normalize(np.array(x_train))
    y_train = np.array(y_train)

    x_test = []
    y_test = []

    for (data, l) in test_data():
        for d in reshape(data):
            x_test.append(d)
            y_test.append(l)

    x_test = normalize(np.array(x_test))
    y_test = np.array(y_test)

    print('Successfully loaded')
    return x_train, y_train, x_test, y_test


def search_hyper_para(x_train, y_train, x_test, y_test):
    tuned_parameters = [{'C': [1, 5, 10, 50, 100, 200, 500, 1000], 'loss': ['hinge', 'squared_hinge']}]

    print('Tuning hyper-parameters for accuracy')
    clf = GridSearchCV(svm.LinearSVC(), tuned_parameters, cv=5, n_jobs=12)
    clf.fit(x_train, y_train)
    print('Best parameters set found on development set:')
    print()
    print(clf.best_params_)
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()


def classify(x_train, y_train, x_test, y_test):
    # train model
    clf = svm.LinearSVC(C=500, loss='hinge')
    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)
    result = {'p_labels': pred, 'test_labels': y_test}
    np.save('result.npyo', result)
    # evaluate the model
    print(clf.score(x_test, y_test))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    classify(x_train, y_train, x_test, y_test)
    # search_hyper_para(x_train, y_train, x_test, y_test)
