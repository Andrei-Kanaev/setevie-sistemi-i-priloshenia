#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №4

# In[14]:


import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def measure_training_time(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    return model, end_time - start_time

def perform_classification(X_train, X_test, y_train, y_test, degrees):
    clf_model = LogisticRegression(max_iter=10000)
    clf_model, clf_time = measure_training_time(clf_model, X_train, y_train)
    clf_accuracy = clf_model.score(X_test, y_test)

    results = []
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test])
        X_poly = poly.fit_transform(X_combined)
        X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
            X_poly, y_combined, test_size=0.3, random_state=42
        )
        clf_poly_model = LogisticRegression(max_iter=10000)
        clf_poly_model, clf_poly_time = measure_training_time(clf_poly_model, X_train_poly, y_train_poly)
        clf_poly_accuracy = clf_poly_model.score(X_test_poly, y_test_poly)
        results.append([degree, clf_poly_accuracy, clf_poly_time])

    return clf_accuracy, clf_time, pd.DataFrame(results, columns=["Degree", "Accuracy", "Training Time (seconds)"])

def perform_regression(X_train, X_test, y_train, y_test, degrees):
    reg_model = LinearRegression()
    reg_model, reg_time = measure_training_time(reg_model, X_train, y_train)
    reg_score = reg_model.score(X_test, y_test)
    coefficients = reg_model.coef_
    features = X_train.columns
    coef_table = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

    results = []
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_combined = pd.concat([X_train, X_test])
        y_combined = pd.concat([y_train, y_test])
        X_poly = poly.fit_transform(X_combined)
        X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
            X_poly, y_combined, test_size=0.3, random_state=42
        )
        reg_model_poly = LinearRegression()
        reg_model_poly, poly_time = measure_training_time(reg_model_poly, X_train_poly, y_train_poly)
        reg_score_poly = reg_model_poly.score(X_test_poly, y_test_poly)
        results.append([degree, reg_score_poly, poly_time])

    return reg_score, reg_time, coef_table, pd.DataFrame(results, columns=["Degree", "R² Score", "Training Time (seconds)"])

url_classification = "https://raw.githubusercontent.com/koroteevmv/ML_course/2023/ML3.1%20polynomial%20features/gen_classification.csv"
data_classification = pd.read_csv(url_classification)
X_classification = data_classification[['x1', 'x2']]
y_classification = data_classification['y']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_classification, y_classification, test_size=0.3, random_state=42
)

degrees = [3, 5, 10]
clf_accuracy, clf_time, classification_results_df = perform_classification(
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, degrees
)

url_house_data = "https://raw.githubusercontent.com/koroteevmv/ML_course/2023/ML3.1%20polynomial%20features/kc_house_data.csv"
data_house = pd.read_csv(url_house_data)
X_house = data_house[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']]
y_house = data_house['price']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_house, y_house, test_size=0.3, random_state=42
)

reg_score, reg_time, coef_table, regression_results_df = perform_regression(
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, degrees
)

print(f"Точность классификации (без полиномов): {clf_accuracy:.2f}")
print(f"Время обучения модели классификации (без полиномов): {clf_time:.4f} секунд")
print("\nРезультаты классификации для разных степеней полиномов:")
print(classification_results_df)
print(f"\nКоэффициент детерминации R² регрессии (без полиномов): {reg_score:.2f}")
print(f"Время обучения модели регрессии (без полиномов): {reg_time:.4f} секунд")
print("\nКоэффициенты линейной модели для предсказания цены дома:")
print(coef_table)
print("\nРезультаты регрессии для разных степеней полиномов:")
print(regression_results_df)

Контрольные вопросы: 
    1) Прогнозирование зависимой переменной от нескольких независимых.
    2) Переобучение, увеличение сложности и потребности в памяти.
    3) Когда данные имеют нелинейные зависимости.
    4) С увеличением степени полинома растет количество признаков.
    5) Для учета всех взаимодействий между признаками.
    6) Линейная регрессия на полиномиальных признаках остается линейной моделью.