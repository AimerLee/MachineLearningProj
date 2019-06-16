import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


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


# lr c-0.01, solver-newton-cg, multi-class-ovr, score mean: 0.27522935779816515, variance: 0.0
# lr c-0.01, solver-newton-cg, multi-class-multinomial, score mean: 0.27522935779816515, variance: 0.0
# lr c-0.01, solver-lbfgs, multi-class-ovr, score mean: 0.27522935779816515, variance: 0.0
# lr c-0.01, solver-lbfgs, multi-class-multinomial, score mean: 0.27522935779816515, variance: 0.0
# lr c-0.01, solver-liblinear, multi-class-ovr, score mean: 0.27522935779816515, variance: 0.0
# lr c-0.1, solver-newton-cg, multi-class-ovr, score mean: 0.28106755629691416, variance: 0.0003067610553971508
# lr c-0.1, solver-newton-cg, multi-class-multinomial, score mean: 0.2929941618015013, variance: 0.0022092430375326654
# lr c-0.1, solver-lbfgs, multi-class-ovr, score mean: 0.28106755629691416, variance: 0.0003067610553971508
# lr c-0.1, solver-lbfgs, multi-class-multinomial, score mean: 0.2929941618015013, variance: 0.0022092430375326654
# lr c-0.1, solver-liblinear, multi-class-ovr, score mean: 0.28106755629691416, variance: 0.0003067610553971508
# lr c-1.0, solver-newton-cg, multi-class-ovr, score mean: 0.287489574645538, variance: 0.002103093973919049
# lr c-1.0, solver-newton-cg, multi-class-multinomial, score mean: 0.278231859883236, variance: 0.003897353994606291
# lr c-1.0, solver-lbfgs, multi-class-ovr, score mean: 0.2876563803169308, variance: 0.002132225840132275
# lr c-1.0, solver-lbfgs, multi-class-multinomial, score mean: 0.2783152627189324, variance: 0.0038910726968053016
# lr c-1.0, solver-liblinear, multi-class-ovr, score mean: 0.28874061718098415, variance: 0.0022227029613919305
# lr c-10, solver-newton-cg, multi-class-ovr, score mean: 0.3122602168473728, variance: 0.004444237309239489
# lr c-10, solver-newton-cg, multi-class-multinomial, score mean: 0.30867389491242697, variance: 0.0036213733852438887
# lr c-10, solver-lbfgs, multi-class-ovr, score mean: 0.31234361968306923, variance: 0.004419202546464559
# lr c-10, solver-lbfgs, multi-class-multinomial, score mean: 0.30892410341951626, variance: 0.0037146885679684412
# lr c-10, solver-liblinear, multi-class-ovr, score mean: 0.3130108423686405, variance: 0.004202007372003776
# lr c-50, solver-newton-cg, multi-class-ovr, score mean: 0.3035863219349458, variance: 0.005298827699758139
# lr c-50, solver-newton-cg, multi-class-multinomial, score mean: 0.31726438698915765, variance: 0.007085999522816136
# lr c-50, solver-lbfgs, multi-class-ovr, score mean: 0.30333611342785655, variance: 0.0052647500940803445
# lr c-50, solver-lbfgs, multi-class-multinomial, score mean: 0.3176814011676397, variance: 0.006904481841623648
# lr c-50, solver-liblinear, multi-class-ovr, score mean: 0.3028356964136781, variance: 0.005173820830675549
# lr c-100, solver-newton-cg, multi-class-ovr, score mean: 0.31092577147623024, variance: 0.00941023274190822
# lr c-100, solver-newton-cg, multi-class-multinomial, score mean: 0.30675562969140946, variance: 0.005589144693138081
# lr c-100, solver-lbfgs, multi-class-ovr, score mean: 0.31092577147623024, variance: 0.009449882130020779
# lr c-100, solver-lbfgs, multi-class-multinomial, score mean: 0.306839032527106, variance: 0.005585617984405965
# lr c-100, solver-liblinear, multi-class-ovr, score mean: 0.30984153461217684, variance: 0.009195910409077348
# lr c-500, solver-newton-cg, multi-class-ovr, score mean: 0.3246038365304421, variance: 0.011206419583737073
# lr c-500, solver-newton-cg, multi-class-multinomial, score mean: 0.32201834862385326, variance: 0.008438704480589539
# lr c-500, solver-lbfgs, multi-class-ovr, score mean: 0.3241034195162636, variance: 0.011050520972091703
# lr c-500, solver-lbfgs, multi-class-multinomial, score mean: 0.32276897414512096, variance: 0.008484829935427143
# lr c-500, solver-liblinear, multi-class-ovr, score mean: 0.31801501251042535, variance: 0.010519893906584648
# lr c-1000, solver-newton-cg, multi-class-ovr, score mean: 0.32385321100917436, variance: 0.005973848098324918
# lr c-1000, solver-newton-cg, multi-class-multinomial, score mean: 0.3140116763969975, variance: 0.007175461063257466
# lr c-1000, solver-lbfgs, multi-class-ovr, score mean: 0.32385321100917436, variance: 0.006004176402214524
# lr c-1000, solver-lbfgs, multi-class-multinomial, score mean: 0.31417848206839033, variance: 0.007212717576017272
# lr c-1000, solver-liblinear, multi-class-ovr, score mean: 0.31843202668890747, variance: 0.005814798403729547
# lr c-5000, solver-newton-cg, multi-class-ovr, score mean: 0.3284403669724771, variance: 0.005190285760791764
# lr c-5000, solver-newton-cg, multi-class-multinomial, score mean: 0.30750625521267716, variance: 0.004593492909367761
# lr c-5000, solver-lbfgs, multi-class-ovr, score mean: 0.32777314428690574, variance: 0.0050626008190033265
# lr c-5000, solver-lbfgs, multi-class-multinomial, score mean: 0.30667222685571305, variance: 0.00443002613381599
# lr c-5000, solver-liblinear, multi-class-ovr, score mean: 0.32593828190158464, variance: 0.004908427303542501
# lr c-10000, solver-newton-cg, multi-class-ovr, score mean: 0.31801501251042535, variance: 0.005646079823261114
# lr c-10000, solver-newton-cg, multi-class-multinomial, score mean: 0.3119266055045872, variance: 0.003710904486015243
# lr c-10000, solver-lbfgs, multi-class-ovr, score mean: 0.31809841534612177, variance: 0.005632188625355715
# lr c-10000, solver-lbfgs, multi-class-multinomial, score mean: 0.3125938281901585, variance: 0.00394404288811708
# lr c-10000, solver-liblinear, multi-class-ovr, score mean: 0.31426188490408674, variance: 0.005084832300478368
# lr c-50000, solver-newton-cg, multi-class-ovr, score mean: 0.31684737281067554, variance: 0.0052988903040551575
# lr c-50000, solver-newton-cg, multi-class-multinomial, score mean: 0.3085904920767306, variance: 0.003154978328479182
# lr c-50000, solver-lbfgs, multi-class-ovr, score mean: 0.3180150125104254, variance: 0.005344466232285593
# lr c-50000, solver-lbfgs, multi-class-multinomial, score mean: 0.3100083402835696, variance: 0.00309349395277271
# lr c-50000, solver-liblinear, multi-class-ovr, score mean: 0.31175979983319435, variance: 0.005213240669699036
# lr c-100000, solver-newton-cg, multi-class-ovr, score mean: 0.3186822351959967, variance: 0.0052559020201015435
# lr c-100000, solver-newton-cg, multi-class-multinomial, score mean: 0.3122602168473728, variance: 0.003067221016123389
# lr c-100000, solver-lbfgs, multi-class-ovr, score mean: 0.3188490408673895, variance: 0.0052675325072812276
# lr c-100000, solver-lbfgs, multi-class-multinomial, score mean: 0.31100917431192654, variance: 0.0031696486020808276
# lr c-100000, solver-liblinear, multi-class-ovr, score mean: 0.3167639699749792, variance: 0.005197798276434144

# lr c-0.01, solver-newton-cg, multi-class-ovr, accuracy: 0.23397435897435898
# lr c-0.01, solver-newton-cg, multi-class-multinomial, accuracy: 0.23397435897435898
# lr c-0.01, solver-lbfgs, multi-class-ovr, accuracy: 0.23397435897435898
# lr c-0.01, solver-lbfgs, multi-class-multinomial, accuracy: 0.23397435897435898
# lr c-0.01, solver-liblinear, multi-class-ovr, accuracy: 0.23397435897435898
# lr c-0.1, solver-newton-cg, multi-class-ovr, accuracy: 0.2362179487179487
# lr c-0.1, solver-newton-cg, multi-class-multinomial, accuracy: 0.24903846153846154
# lr c-0.1, solver-lbfgs, multi-class-ovr, accuracy: 0.23573717948717948
# lr c-0.1, solver-lbfgs, multi-class-multinomial, accuracy: 0.24919871794871795
# lr c-0.1, solver-liblinear, multi-class-ovr, accuracy: 0.23557692307692307
# lr c-1.0, solver-newton-cg, multi-class-ovr, accuracy: 0.2987179487179487
# lr c-1.0, solver-newton-cg, multi-class-multinomial, accuracy: 0.308974358974359
# lr c-1.0, solver-lbfgs, multi-class-ovr, accuracy: 0.2988782051282051
# lr c-1.0, solver-lbfgs, multi-class-multinomial, accuracy: 0.308974358974359
# lr c-1.0, solver-liblinear, multi-class-ovr, accuracy: 0.2988782051282051
# lr c-10, solver-newton-cg, multi-class-ovr, accuracy: 0.39791666666666664
# lr c-10, solver-newton-cg, multi-class-multinomial, accuracy: 0.4394230769230769
# lr c-10, solver-lbfgs, multi-class-ovr, accuracy: 0.3974358974358974
# lr c-10, solver-lbfgs, multi-class-multinomial, accuracy: 0.43846153846153846
# lr c-10, solver-liblinear, multi-class-ovr, accuracy: 0.39823717948717946
# lr c-50, solver-newton-cg, multi-class-ovr, accuracy: 0.5610576923076923
# lr c-50, solver-newton-cg, multi-class-multinomial, accuracy: 0.5897435897435898
# lr c-50, solver-lbfgs, multi-class-ovr, accuracy: 0.5612179487179487
# lr c-50, solver-lbfgs, multi-class-multinomial, accuracy: 0.5892628205128205
# lr c-50, solver-liblinear, multi-class-ovr, accuracy: 0.5631410256410256
# lr c-100, solver-newton-cg, multi-class-ovr, accuracy: 0.60625
# lr c-100, solver-newton-cg, multi-class-multinomial, accuracy: 0.6243589743589744
# lr c-100, solver-lbfgs, multi-class-ovr, accuracy: 0.6064102564102564
# lr c-100, solver-lbfgs, multi-class-multinomial, accuracy: 0.6246794871794872
# lr c-100, solver-liblinear, multi-class-ovr, accuracy: 0.6067307692307692
# lr c-500, solver-newton-cg, multi-class-ovr, accuracy: 0.6711538461538461
# lr c-500, solver-newton-cg, multi-class-multinomial, accuracy: 0.670352564102564
# lr c-500, solver-lbfgs, multi-class-ovr, accuracy: 0.6711538461538461
# lr c-500, solver-lbfgs, multi-class-multinomial, accuracy: 0.6743589743589744
# lr c-500, solver-liblinear, multi-class-ovr, accuracy: 0.6751602564102565
# lr c-1000, solver-newton-cg, multi-class-ovr, accuracy: 0.6849358974358974
# lr c-1000, solver-newton-cg, multi-class-multinomial, accuracy: 0.6727564102564103
# lr c-1000, solver-lbfgs, multi-class-ovr, accuracy: 0.6857371794871795
# lr c-1000, solver-lbfgs, multi-class-multinomial, accuracy: 0.6730769230769231
# lr c-1000, solver-liblinear, multi-class-ovr, accuracy: 0.6891025641025641
# lr c-5000, solver-newton-cg, multi-class-ovr, accuracy: 0.6899038461538461
# lr c-5000, solver-newton-cg, multi-class-multinomial, accuracy: 0.6652243589743589
# lr c-5000, solver-lbfgs, multi-class-ovr, accuracy: 0.690224358974359
# lr c-5000, solver-lbfgs, multi-class-multinomial, accuracy: 0.6644230769230769
# lr c-5000, solver-liblinear, multi-class-ovr, accuracy: 0.6969551282051282
# lr c-10000, solver-newton-cg, multi-class-ovr, accuracy: 0.6919871794871795
# lr c-10000, solver-newton-cg, multi-class-multinomial, accuracy: 0.6613782051282051
# lr c-10000, solver-lbfgs, multi-class-ovr, accuracy: 0.6943910256410256
# lr c-10000, solver-lbfgs, multi-class-multinomial, accuracy: 0.6620192307692307
# lr c-10000, solver-liblinear, multi-class-ovr, accuracy: 0.6998397435897435         (Best)
# lr c-50000, solver-newton-cg, multi-class-ovr, accuracy: 0.6911858974358974
# lr c-50000, solver-newton-cg, multi-class-multinomial, accuracy: 0.6567307692307692
# lr c-50000, solver-lbfgs, multi-class-ovr, accuracy: 0.6915064102564102
# lr c-50000, solver-lbfgs, multi-class-multinomial, accuracy: 0.6567307692307692
# lr c-50000, solver-liblinear, multi-class-ovr, accuracy: 0.6927884615384615
# lr c-100000, solver-newton-cg, multi-class-ovr, accuracy: 0.6873397435897436
# lr c-100000, solver-newton-cg, multi-class-multinomial, accuracy: 0.6540064102564103
# lr c-100000, solver-lbfgs, multi-class-ovr, accuracy: 0.6886217948717949
# lr c-100000, solver-lbfgs, multi-class-multinomial, accuracy: 0.6544871794871795
# lr c-100000, solver-liblinear, multi-class-ovr, accuracy: 0.6858974358974359


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    # be free to use data below
    # C=1.0, random_state=0, solver=newton-cg/lbfgs/liblinear, max_iter=100, multi_class=‘ovr’/‘multinomial’
    # for C in [0.01, 0.1, 1.0, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
    # for C in [50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
    #     for solver in ['newton-cg', 'lbfgs', 'liblinear']:
    #         for multi_class in ['ovr', 'multinomial']:
    #             if solver=='liblinear' and multi_class=='multinomial':
    #                 break
    #             clf = LogisticRegression(C=C, random_state=0, solver=solver, max_iter=10000, multi_class=multi_class)
    #             scores = cross_val_score(clf, x_train, y_train, cv=10, n_jobs=-1)
    #             print('lr c-{}, solver-{}, multi-class-{}, score mean: {}, variance: {}'.format(C, solver, multi_class, scores.mean(), scores.std() ** 2))
    #             # clf.fit(x_train, y_train)
    #             # pred_test = clf.predict(x_test)
    #             # right_pred = 0
    #             # for i in range(len(y_test)):
    #             #     right_pred += 1 if (pred_test[i] == y_test[i]) else 0
    #             # print('lr c-{}, solver-{}, multi-class-{}, accuracy: {}'.format(C, solver, multi_class, right_pred/len(y_test)))

    clf = LogisticRegression(C=10000, random_state=0, solver='liblinear', max_iter=10000, multi_class='ovr')
    clf.fit(x_train, y_train)
    pred_test = clf.predict(x_test)
    right_pred = 0
    for i in range(len(y_test)):
        right_pred += 1 if (pred_test[i] == y_test[i]) else 0
    print('accuracy: {}'.format(right_pred / len(y_test)))
    # np.save('lr_pred.npy', pred_test)
    # np.save('lr_test.npy', y_test)
