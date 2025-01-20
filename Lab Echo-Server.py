#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №3

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, class_sep=2, random_state=1)

plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], marker="o", c='r', s=100, label="Класс 0")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], marker="x", c='b', s=100, label="Класс 1")
plt.legend()
plt.show()

class GradientDescent:
    def __init__(self, learning_rate=0.5, iterations=1000):
        self.weights = None
        self.learning_rate = learning_rate
        self.iterations = iterations

    def step(self, weights, grad):
        return weights - self.learning_rate * grad

    def optimize(self, features, target, initial_weights):
        weights = initial_weights.copy()
        for _ in range(self.iterations):
            grad = self.compute_gradient(features, target, weights)
            weights = self.step(weights, grad)
        return weights

    def fit(self, features, target):
        initial_weights = np.ones(features.shape[1])
        self.weights = self.optimize(features, target, initial_weights)

class CustomLogisticRegression(GradientDescent):
    def sigmoid(self, features, weights):
        return 1 / (1 + np.exp(-features.dot(weights)))

    def compute_gradient(self, features, target, weights):
        n = features.shape[0]
        return (1 / n) * features.T.dot(self.sigmoid(features, weights) - target)

    def predict_proba(self, features):
        return self.sigmoid(features, self.weights)

    def predict(self, features):
        return self.predict_proba(features) > 0.5

X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

custom_model = CustomLogisticRegression(learning_rate=0.5, iterations=1000)
custom_model.fit(X_bias, y)
predictions = custom_model.predict(X_bias)

accuracy_custom = accuracy_score(y, predictions)
f1_custom = f1_score(y, predictions)
print(f'Точность = {accuracy_custom:.2f}, F1-мера = {f1_custom:.2f}')

xx, yy = np.meshgrid(
    np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.01), 
    np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.01))

grid_points = np.c_[np.ones((xx.size, 1)), xx.ravel(), yy.ravel()]
Z = custom_model.predict_proba(grid_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], marker="o", c='r', s=100, label="Класс 0")
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], marker="x", c='b', s=100, label="Класс 1")
plt.legend()
plt.show()

sklearn_model = LogisticRegression()
sklearn_model.fit(X_bias, y)
predictions_sklearn = sklearn_model.predict(X_bias)

accuracy_sklearn = accuracy_score(y, predictions_sklearn)
f1_sklearn = f1_score(y, predictions_sklearn)
print(f'Точность логистической регрессии Sklearn = {accuracy_sklearn:.2f}, F1-мера = {f1_sklearn:.2f}')

Контрольные вопросы:
    1) Задача классификации — разделить объекты на классы по характеристикам. Примеры: определение спама, диагностика              заболеваний, классификация изображений.
    2) Шаг градиентного спуска — параметр, определяющий, насколько сильно корректируются параметры модели на каждом шаге          обучения.
    3) Функция ошибки в логистической регрессии — логистическая функция потерь или кросс-энтропия.
    4) Столбец единиц добавляется для учета свободного члена (сдвига), что улучшает качество разделения классов.