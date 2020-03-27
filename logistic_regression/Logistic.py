import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

from Utils import k_fold_cross_validation, f_measure


CLASSES_NUMBER = 2


def read_dataset(path):
	data = pd.read_csv(path)
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].apply(lambda c: 1 if c == 'P' else -1).values
	return X, y


class Logistic:
	def __init__(self, lmb, learning_rate, amount_steps=10000, eps=1e-5):
		self.lmb = lmb
		self.learning_rate = learning_rate
		self.amount_steps = amount_steps
		self.eps = eps
		self.w = None

	def __init_weight(self, m):
		self.w = np.random.normal(loc=0., scale=1., size=m)

	def __evaluate_Q(self, X, y, n):
		losses = np.sum([np.logaddexp(0, -y[i] * np.dot(X[i], self.w)) for i in range(n)])
		regularization = self.lmb / 2 * np.sum(self.w ** 2)
		return (losses / n) + regularization

	def fit(self, X, y):
		n, m = X.shape
		self.__init_weight(m)
		Q = self.__evaluate_Q(X, y, n)
		for s in range(self.amount_steps):
			gradient = 0
			for i in range(n):
				gradient += expit(-y[i] * np.dot(X[i], self.w)) * np.sum([y[i] * X[i][j] for j in range(m)])
			gradient *= -1 / n
			gradient += self.lmb * np.sum(self.w)
			self.w = self.w - self.learning_rate * gradient

			new_Q = self.__evaluate_Q(X, y, n)
			if np.linalg.norm(new_Q - Q) < self.eps:
				break
			Q = new_Q

	def predict(self, X):
		return [np.sign(np.dot(x_i, self.w)) for x_i in X]


def logistic(train_X, train_y, test_X, test_y, lmb, learning_rate):
	alg = Logistic(lmb, learning_rate)
	alg.fit(train_X, train_y)
	predict_result = alg.predict(test_X)

	confusion_matrix = [[0] * CLASSES_NUMBER for _ in range(CLASSES_NUMBER)]
	cm_map = {-1.0: 0, 1.0: 1}
	for (clf_ans, ans) in zip(predict_result, test_y):
		confusion_matrix[cm_map.get(clf_ans)][cm_map.get(ans)] += 1
	return f_measure(confusion_matrix, CLASSES_NUMBER)


def train_dataset(X, y, lmb, learning_rate):
	k = 5
	amount_f = 0
	for [train_X, train_y, test_X, test_y] in k_fold_cross_validation(k, X, y, CLASSES_NUMBER):
		amount_f += logistic(train_X, train_y, test_X, test_y, lmb, learning_rate)

	avg_f = amount_f / k
	return avg_f


def make_plot(X, y, lmb, learning_rate):
	clf = Logistic(lmb, learning_rate)
	clf.fit(X, y)
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()]))
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
	plt.title("Logistic")
	plt.axis('tight')
	plt.show()


if __name__ == '__main__':
	X, y = read_dataset('data/geyser.csv')
	best_F = -1
	best_lmb = None
	best_rate = None
	for lmb in [0.0001, 0.001, 0.01, 0.1, 1.]:
		for learning_rate in [1e-3, 1e-8]:
			f_score = train_dataset(X, y, lmb, learning_rate)
			if f_score > best_F:
				best_F = f_score
				best_lmb = lmb
				best_rate = learning_rate
	print(f'best F-measure: {best_F}, lambda: {best_lmb}, rate: {best_rate}')
	#make_plot(X, y, 0.01, 0.001)
