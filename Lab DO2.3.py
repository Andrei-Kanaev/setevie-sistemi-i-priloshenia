#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №2

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    x = pd.read_csv('https://raw.githubusercontent.com/koroteevmv/ML_course/2023/ML1.1%20linear%20regression/data/x.csv', index_col=0)['0']
    y = pd.read_csv('https://raw.githubusercontent.com/koroteevmv/ML_course/2023/ML1.1%20linear%20regression/data/y.csv', index_col=0)['0']
    return x, y

class Model:
    def __init__(self, b0=0, b1=0):
        self.b0 = b0
        self.b1 = b1

    def predict(self, X):
        return self.b0 + self.b1 * X

    def error(self, X, Y):
        return sum(((self.predict(X) - Y) ** 2) / (2 * len(X)))

    def fit(self, X, Y, alpha=0.001, tol=1e-6, max_steps=10000, adaptive=False):
        steps, errors = [], []
        for step in range(max_steps):
            dJ0 = sum(self.predict(X) - Y) / len(X)
            dJ1 = sum((self.predict(X) - Y) * X) / len(X)
            self.b0 -= alpha * dJ0
            self.b1 -= alpha * dJ1
            new_error = self.error(X, Y)
            errors.append(new_error)
            steps.append(step)
            if step > 0 and abs(errors[-1] - errors[-2]) < tol:
                break
            if adaptive and step > 0 and errors[-1] > errors[-2]:
                alpha /= 2
        return steps, errors

    def plot_regression(self, X, Y):
        plt.figure()
        plt.scatter(X, Y, label='Data points')
        X_vals = np.linspace(min(X), max(X), 100)
        plt.plot(X_vals, self.predict(X_vals), 'r', label='Regression line')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def plot_learning_curve(self, steps, errors):
        plt.figure()
        plt.plot(steps, errors, 'g', label='Learning curve')
        plt.xlabel("Steps")
        plt.ylabel("Error")
        plt.legend()
        plt.show()

def main():
    x, y = load_data()
    model = Model()
    steps, errors = model.fit(x, y, alpha=0.01, adaptive=True)
    print("Финальная ошибка после градиентного спуска:", errors[-1])
    model.plot_regression(x, y)
    model.plot_learning_curve(steps, errors)

if __name__ == "__main__":
    main()

Контрольные вопросы: 
    1) Задача регрессии: Определение зависимости между независимыми переменными и непрерывной зависимой переменной. 
       Примеры: прогнозирование цен на недвижимость, температура, рост растений.
    2) Метод градиентного спуска: Алгоритм оптимизации, минимизирующий функцию ошибки, двигаясь по направлению                    антиградиента. 
    3) Скорость обучения: Параметр, определяющий шаг обновления модели на каждом шаге алгоритма. Влияет на скорость                сходимости.
    4) Функции ошибки: Среднеквадратичная ошибка (MSE) и средняя абсолютная ошибка (MAE). Используются для оценки точности        предсказаний.
    5) Значение ошибки регрессии: Показывает точность предсказаний модели. Меньше — лучше.
    6) График обучения: Показывает изменение ошибки на протяжении обучения. Помогает оценить эффективность модели и                настройку параметров.