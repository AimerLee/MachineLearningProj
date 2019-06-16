import pickle
from itertools import permutations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from tinydb import where
from sklearn.metrics import accuracy_score
from BCMI.constant import ATTRS
from BCMI.dataset import SEED
from dataset import CustomResolver

LABELS = [[4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
          [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
          [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]]


def load_dataset(rows):
    """Load Dataset根据不同的需要可以进行不同的定制，比如不同的normalization方式可以实现不同的load_dataset的方式"""
    dataset = []
    labels = []
    for row in rows:
        label = row[ATTRS.LABEL]
        data = np.load(row[ATTRS.DATA_PATH])

        dataset.append(data)
        labels += [label] * data.shape[2]
    dataset = np.concatenate(dataset, axis=2)
    labels = np.array(labels)

    dataset = dataset.reshape(310, labels.shape[0]).T  # (N,310)
    dataset = normalize(dataset)
    return dataset, labels


def validation(dataset):
    blocks = [dataset.search(where(ATTRS.TRIAL_INDEX) < 5),
              dataset.search((where(ATTRS.TRIAL_INDEX) < 10) & (where(ATTRS.TRIAL_INDEX) >= 5))]

    validate_set_acc = []
    for train_rows, validate_rows in permutations(blocks):
        train_dataset, train_labels = load_dataset(train_rows)
        validate_dataset, validate_labels = load_dataset(validate_rows)

        acc_list = []
        for n in np.arange(1, 100, 2):
            classifier = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
            classifier.fit(train_dataset, train_labels)
            acc = classifier.score(validate_dataset, validate_labels)
            acc_list.append(acc)
            print("Validation Acc, N={:d} : {:.2f}".format(n, acc))

        validate_set_acc.append(acc_list)
    validate_set_acc = np.array(validate_set_acc)
    return validate_set_acc


def train_knn_with_validate(dataset):
    # Get best n
    acc_array = validation(dataset)
    np.save('validate_acc.npy', acc_array)
    n = 1 + 2 * acc_array.mean(axis=0).argmax()
    print("Best n for KNN: {:d}".format(n))

    # Training
    train_rows = dataset.search(where(ATTRS.TRIAL_INDEX) < 10)
    test_rows = dataset.search(where(ATTRS.TRIAL_INDEX) >= 10)

    train_dataset, train_labels = load_dataset(train_rows)
    test_dataset, test_labels = load_dataset(test_rows)

    classifier = KNeighborsClassifier(n_neighbors=n, n_jobs=-1)
    classifier.fit(train_dataset, train_labels)
    p_labels = classifier.predict(test_dataset)

    print("Final Acc: {:.3f}".format(accuracy_score(test_labels, p_labels)))
    result = {'p_labels': p_labels, 'test_labels': test_labels}
    with open('knn_result.npy', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    # Load Dataset
    dataset = SEED()
    dataset.purge()  # Reset the dataset every time the program runs
    dataset.index_dataset(CustomResolver(dataset_dir='./data/dataset', labels=LABELS))

    train_knn_with_validate(dataset)
