import numpy as np
from copy import copy


def safe_div(a, b):
    return 0 if b == 0 else a / b


def f_measure(confusion_matrix, amount_classes):
    precision = 0
    recall = 0
    for i in range(amount_classes):
        precision += safe_div(confusion_matrix[i][i], sum(confusion_matrix[i]))
        recall += safe_div(confusion_matrix[i][i], sum(np.array(confusion_matrix).T[i]))
    precision /= amount_classes
    recall /= amount_classes
    return safe_div(2 * recall * precision, recall + precision)


def k_fold_cross_validation(k, X, y, amount_classes=2):
    n, m = X.shape
    buckets = [[] for _ in range(amount_classes)]
    parts = [[] for _ in range(k)]
    for i in range(n):
        buckets[0].append([X[i], y[i]]) if y[i] > 0 else buckets[1].append([X[i], y[i]])
    buckets.sort(key=lambda arr: len(arr), reverse=True)
    cur_k = 0
    for i in range(amount_classes):
        for el in buckets[i]:
            parts[cur_k].append(el)
            cur_k = (cur_k + 1) % k
    samples = []
    for i in range(k):
        my_set = set(range(k))
        my_set.remove(i)
        train_sample = []
        test_sample = copy(parts[i])
        for ind in my_set:
            train_sample += parts[ind]
        train_x, train_y = [], []

        for train_elem in train_sample:
            train_x.append(train_elem[0])
            train_y.append(train_elem[1])
        test_x, test_y = [], []
        for test_elem in test_sample:
            test_x.append(test_elem[0])
            test_y.append(test_elem[1])
        samples.append([np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)])
    return samples
