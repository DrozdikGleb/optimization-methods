import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve

from Utils import k_fold_cross_validation, f_measure

CLASSES_NUMBER = 2


def read_dataset(path):
	data = pd.read_csv(path)
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].apply(lambda c: 1 if c == 'P' else -1).values
	return X, y


class Logistic:
	def __init__(self, lmb, learning_rate, amount_steps=100000, eps=1e-5):
		self.lmb = lmb
		self.learning_rate = learning_rate
		self.amount_steps = amount_steps
		self.eps = eps
		self.w = None

	def __init_weight(self, m):
		self.w = np.random.normal(loc=0., scale=1., size=m)

	def __evaluate_Q(self, X, y, n):
		predictions = np.matmul(X, self.w)
		margins = predictions * y
		losses = np.logaddexp(0, -margins)
		return 1 / n * losses + self.lmb / 2 * np.sum(np.square(self.w))

	def __evaluate_gradient_Q(self, X, y, n):
		predictions = np.matmul(X, self.w)
		margins = predictions * y
		b = expit(-margins)
		A = np.transpose(X * y.reshape((n, 1)))
		return -1 / n * np.matmul(A, b) + self.lmb * self.w

	def __evaluate_hessian_Q(self, X, y, n, m):
		predictions = np.matmul(X, self.w)
		margins = predictions * y
		C = np.transpose(X * expit(-margins).reshape((n, 1)))
		D = X * expit(margins).reshape((n, 1))
		return 1 / n * np.matmul(C, D) + self.lmb * np.eye(m)

	def fit(self, X, y, solution_type):
		n, m = X.shape
		self.__init_weight(m)
		Q = self.__evaluate_Q(X, y, n)
		for s in range(self.amount_steps):
			if solution_type == 'gradient':
				self.w = self.w - self.learning_rate * self.__evaluate_gradient_Q(X, y, n)
				new_Q = self.__evaluate_Q(X, y, n)
				if np.linalg.norm(new_Q - Q) < self.eps:
					break
				Q = new_Q
			elif solution_type == 'newton':
				hess = self.__evaluate_hessian_Q(X, y, n, m)
				hess_inv = cho_solve(cho_factor(hess), np.eye(hess.shape[0]))
				delta = np.matmul(self.__evaluate_gradient_Q(X, y, n), hess_inv)
				self.w = self.w - delta
				if np.linalg.norm(delta) < self.eps:
					break
			else:
				print(f'This type of solution {solution_type} is not supported')

	def predict(self, X):
		return [np.sign(np.dot(x_i, self.w)) for x_i in X]


def logistic(train_X, train_y, test_X, test_y, lmb, learning_rate, solution_type):
	cls = Logistic(lmb, learning_rate)
	cls.fit(train_X, train_y, solution_type)
	predict_result = cls.predict(test_X)

	confusion_matrix = [[0] * CLASSES_NUMBER for _ in range(CLASSES_NUMBER)]
	cm_map = {-1.0: 0, 1.0: 1}
	for (clf_ans, ans) in zip(predict_result, test_y):
		confusion_matrix[cm_map.get(clf_ans)][cm_map.get(ans)] += 1
	return f_measure(confusion_matrix, CLASSES_NUMBER)


def train_dataset(X, y, lmb, learning_rate, solution_type):
	k = 5
	amount_f = 0
	for [train_X, train_y, test_X, test_y] in k_fold_cross_validation(k, X, y, CLASSES_NUMBER):
		amount_f += logistic(train_X, train_y, test_X, test_y, lmb, learning_rate, solution_type)

	avg_f = amount_f / k
	return avg_f


def make_plot(cls, X, y, solution_type):
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = np.array(cls.predict(np.c_[xx.ravel(), yy.ravel()]))
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
	plt.title(f'Logistic with {solution_type}')
	plt.axis('tight')
	plt.show()


def find_best(X, y, solution_type):
	best_F = -1
	best_lmb = None
	best_rate = None
	for lmb in [1e-4, 1e-3, 1e-2, 1e-1, 1.]:
		for learning_rate in [1e-3, 1e-5, 1e-8]:
			f_score = train_dataset(X, y, lmb, learning_rate, solution_type)
			if f_score > best_F:
				best_F = f_score
				best_lmb = lmb
				best_rate = learning_rate

	print(f'best F-measure with {solution_type}: {best_F}, lambda: {best_lmb}, learning_rate: {best_rate}')


if __name__ == '__main__':
	X, y = read_dataset('../data/geyser.csv')
	find_best(X, y, 'gradient')
	find_best(X, y, 'newton')