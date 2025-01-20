#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа 

# # Вариант: debutanizer
ВЫПОЛНИЛ: Канаев Андрей, ДПИ22-1


# In[2]:


# Импорт необходимых библиотек
import numpy as np  # NumPy для работы с массивами
import matplotlib.pyplot as plt  # Matplotlib для визуализации данных
from sklearn.datasets import fetch_openml  # Импорт функции для загрузки датасета
from sklearn.linear_model import LinearRegression  # Импорт линейной регрессии
from sklearn.metrics import r2_score, mean_squared_error  # Импорт метрик для оценки модели

# Загрузка датасета 'debutanizer'
dataset = fetch_openml(name='debutanizer', version=1, as_frame=True)
y = dataset.target  # Определение целевой переменной
X = dataset.data  # Определение признаков

# Исследование датасета
print("Число строк (объектов):", X.shape[0])
print("Число столбцов (признаков):", X.shape[1])
print(X.describe())
print()

print("Типы данных признаков:\n", X.dtypes)
print("Тип данных целевой переменной:", y.dtype)

# Обработка пропущенных значений и замена их медианными значениями
print("Пропущенные значения в признаках:", X.isnull().sum().sum())
print("Пропущенные значения в целевой переменной:", y.isnull().sum())
X = X.fillna(X.median())

# Визуализация распределения целевой переменной
plt.hist(y, bins=20, edgecolor='black')
plt.title("Распределение целевой переменной")
plt.xlabel("Значения целевой переменной")
plt.ylabel("Частота")
plt.show()
print("Выводы по гистограмме распределения целевой переменной:")

# Реализация линейной регрессии
def linear_regression(X, y, learning_rate=0.01, epochs=1000):
    # Инициализация переменной weights нулями и установка начального смещения равного 0
    weights = np.zeros(X.shape[1])
    bias = 0
    errors = []  # Список для хранения значений ошибки на каждой итерации обучения
    
    # Цикл обучения модели
    for epoch in range(epochs):
        # Прогнозирование значений 
        predictions = np.dot(X, weights) + bias
        
        # Вычисление ошибки и добавление её в список
        errors.append(mean_squared_error(y, predictions))
        
        # Обновление weights с использованием градиентного спуска
        weights -= learning_rate * (1/X.shape[0]) * np.dot(X.T, predictions - y)
        
        # Обновление смещения с использованием градиентного спуска
        bias -= learning_rate * (1/X.shape[0]) * np.sum(predictions - y)
    
    # Возврат значений
    return weights, bias, errors

# Добавление столбца для bias в матрицу признаков
X['bias'] = 1
weights, bias, errors = linear_regression(X, y)

# Визуализация процесса обучения
plt.plot(range(1, 1001), errors)
plt.title("График обучения модели линейной регрессии")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.show()

# Визуализация распределения целевой переменной и предсказаний модели
plt.scatter(y, np.dot(X, weights) + bias)
plt.title("Распределение целевой переменной и предсказаний модели")
plt.xlabel("Целевая переменная")
plt.ylabel("Предсказания модели")
plt.show()

# Отображение уравнения гиперплоскости для созданной модели
equation = "Уравнение гиперплоскости: "
for i in range(len(weights)):
    equation += f"{weights[i]:.2f} * {X.columns[i]} + "
equation += f"{bias:.2f}"
print(equation)

# Обучение линейной регрессии с использованием sklearn
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

# Отображение уравнения гиперплоскости для модели sklearn
equation_sklearn = "Уравнение гиперплоскости (sklearn): "
for i in range(len(model_sklearn.coef_)):
    equation_sklearn += f"{model_sklearn.coef_[i]:.2f} * {X.columns[i]} + "
equation_sklearn += f"{model_sklearn.intercept_:.2f}"
print(equation_sklearn)

# Вычисление прогнозов модели, обученной вручную (predictions_custom)
predictions_custom = np.dot(X, weights) + bias
# Вычисление коэффициента детерминации (R^2) для модели, обученной вручную (r2_custom)
r2_custom = r2_score(y, predictions_custom)
# Вычисление среднеквадратичной ошибки (MSE) для модели, обученной вручную (mse_custom)
mse_custom = mean_squared_error(y, predictions_custom)
# Вычисление прогнозов модели, обученной с использованием библиотеки sklearn (predictions_sklearn)
predictions_sklearn = model_sklearn.predict(X)
# Вычисление коэффициента детерминации (R^2) для модели, обученной с использованием библиотеки sklearn (r2_sklearn)
r2_sklearn = r2_score(y, predictions_sklearn)
# Вычисление среднеквадратичной ошибки (MSE) для модели, обученной с использованием библиотеки sklearn (mse_sklearn)
mse_sklearn = mean_squared_error(y, predictions_sklearn)

print("Метрики модели своими руками:")
print(f"Коэффициент детерминации (R^2): {r2_custom:.4f}")
print(f"Ошибка MSE: {mse_custom:.4f}")

print("\nМетрики модели sklearn:")
print(f"Коэффициент детерминации (R^2): {r2_sklearn:.4f}")
print(f"Ошибка MSE: {mse_sklearn:.4f}")

