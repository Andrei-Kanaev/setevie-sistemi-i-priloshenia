#!/usr/bin/env python
# coding: utf-8

# # Контрольная работа №2. Классификация

# # Вариант: diggle_table_a2
ВЫПОЛНИЛ:  Канаев Андрей, ДПИ22-1


# In[2]:


# Импорт библиотек
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
import time
import pandas as pd

# Загрузка данных из OpenML
data = fetch_openml(name='diggle_table_a2', version=1, as_frame=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.data
y = data.target

# Объединение признаков и целевой переменной в один DataFrame
df = pd.concat([X, y], axis=1)

# Вывод информации о размере и описании данных
print("Число строк и столбцов:", df.shape)
print("\nОписание данных:")
print(df.describe())
print("\nУникальные значения целевой переменной:")
print(df[y.name].value_counts())

# Проверка наличия пропущенных значений и их удаление, если они есть
if df.isnull().sum().sum() == 0:
    print("\nПропущенных значений нет.")
else:
    df = df.dropna()

# Приведение типов данных к float64, если все признаки численные
if df.dtypes.value_counts().shape[0] == 1 and df.dtypes.unique()[0] in [float, int]:
    print("\nВсе признаки численные.")
else:
    df = df.astype('float64')

# Разделение данных на обучающую и тестовую выборки
X = df.drop(columns=[y.name])
y = df[y.name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание экземпляра модели логистической регрессии с заданными параметрами
log_reg_model = LogisticRegression(max_iter=100000, solver='saga', random_state=42, warm_start=True)
# Засекаем время перед началом обучения модели
start_time = time.time()
# Обучение модели на обучающих данных
log_reg_model.fit(X_train, y_train)
# Засекаем время после завершения обучения
end_time = time.time()
# Предсказание меток классов на тестовых данных
y_pred = log_reg_model.predict(X_test)
# Вычисление точности модели с использованием accuracy_score
accuracy_log_reg = accuracy_score(y_test, y_pred)
# Генерация отчета о классификации с метриками precision, recall, f1-score и support
classification_report_log_reg = classification_report(y_test, y_pred)

print("\nМетрики логистической регрессии:")
print(f"Accuracy: {accuracy_log_reg:.4f}")
print("\nТаблица классификации:")
print(classification_report_log_reg)
print(f"Время обучения: {end_time - start_time:.4f} сек")

# Поиск наилучшей степени полинома для полиномиальной модели
# Список степеней полиномов, которые будут опробованы
degrees = [2, 3, 4]

# Инициализация переменных для хранения лучшей степени полинома и соответствующей точности
best_degree = None
best_accuracy_poly = 0

# Цикл по всем степеням полиномов
for degree in degrees:
    # Создание объекта PolynomialFeatures для преобразования данных
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Разделение полиномиальных данных на обучающую и тестовую выборки
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Создание и обучение модели логистической регрессии на полиномиальных данных
    log_reg_poly_model = LogisticRegression()
    start_time_poly = time.time()
    log_reg_poly_model.fit(X_train_poly, y_train_poly)
    end_time_poly = time.time()

    # Предсказание меток классов на тестовых данных
    y_pred_poly = log_reg_poly_model.predict(X_test_poly)

    # Вычисление точности модели с использованием accuracy_score
    accuracy_poly = accuracy_score(y_test_poly, y_pred_poly)

    # Обновление лучшей степени полинома и соответствующей точности, если текущая точность лучше
    if accuracy_poly > best_accuracy_poly:
        best_accuracy_poly = accuracy_poly
        best_degree = degree


print("\nЛучшая степень полинома:", best_degree)
print(f"Лучшая точность: {best_accuracy_poly:.4f}")

# Список ядер SVM, которые будут опробованы
kernels = ['linear', 'poly', 'rbf']

# Инициализация переменных для хранения лучшего ядра и соответствующей точности
best_kernel = None
best_accuracy_svm = 0

# Цикл по всем ядрам SVM
for kernel in kernels:
    # Создание объекта SVM с заданным ядром
    svm_model = SVC(kernel=kernel)
    # Засекаем время перед началом обучения модели
    start_time_svm = time.time()
    # Обучение модели SVM на обучающих данных
    svm_model.fit(X_train, y_train)
    # Засекаем время после завершения обучения
    end_time_svm = time.time()
    # Предсказание меток классов на тестовых данных
    y_pred_svm = svm_model.predict(X_test)
    # Вычисление точности модели с использованием accuracy_score
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    # Обновление лучшего ядра и соответствующей точности, если текущая точность лучше
    if accuracy_svm > best_accuracy_svm:
        best_accuracy_svm = accuracy_svm
        best_kernel = kernel

print("\nЛучшее ядро SVM:", best_kernel)
print(f"Лучшая точность: {best_accuracy_svm:.4f}")

# Создание объекта модели Перцептрона
perceptron_model = Perceptron()
# Засекаем время перед началом обучения модели Перцептрона
start_time_perceptron = time.time()
# Обучение модели Перцептрона на обучающих данных
perceptron_model.fit(X_train, y_train)
# Засекаем время после завершения обучения
end_time_perceptron = time.time()
# Предсказание меток классов на тестовых данных
y_pred_perceptron = perceptron_model.predict(X_test)
# Вычисление точности модели Перцептрона с использованием accuracy_score
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
# Генерация отчета о классификации для модели Перцептрона
classification_report_perceptron = classification_report(y_test, y_pred_perceptron)


# Вывод метрик модели Перцептрона
print("\nМетрики Перцептрона:")
print(f"Accuracy: {accuracy_perceptron:.4f}")
print("\nТаблица классификации:")
print(classification_report_perceptron)
print(f"Время обучения: {end_time_perceptron - start_time_perceptron:.4f} сек")

# Создание DataFrame с результатами
results_data = {
    'Модель': ['Логистическая регрессия', 'Полиномиальная модель', 'SVM', 'Перцептрон'],
    'Точность': [accuracy_log_reg, best_accuracy_poly, best_accuracy_svm, accuracy_perceptron],
    'Время обучения (сек)': [end_time - start_time, end_time_poly - start_time_poly, end_time_svm - start_time_svm,
                             end_time_perceptron - start_time_perceptron]
}

results_df = pd.DataFrame(results_data)
print("\nИтоговая таблица:")
print(results_df)

