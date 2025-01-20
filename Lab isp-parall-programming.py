#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №5

# In[12]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Все колонки, кроме последней, как признаки
    y = data.iloc[:, -1].values   # Последняя колонка как целевая переменная
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def initialize_model():
    return LogisticRegression(max_iter=5000)

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict(model, X):
    return model.predict(X)

def evaluate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def cross_validate(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train_stratified, X_test_stratified = X[train_index], X[test_index]
        y_train_stratified, y_test_stratified = y[train_index], y[test_index]
        
        train_model(model, X_train_stratified, y_train_stratified)
        
        y_train_pred = predict(model, X_train_stratified)
        y_test_pred = predict(model, X_test_stratified)
        
        train_accuracy, train_precision, train_recall, train_f1 = evaluate_metrics(y_train_stratified, y_train_pred)
        test_accuracy, test_precision, test_recall, test_f1 = evaluate_metrics(y_test_stratified, y_test_pred)
        
        print("\nСтратифицированное разделение (кросс-валидация):")
        print(f"Точность на обучающей выборке: {train_accuracy}")
        print(f"Точность на тестовой выборке: {test_accuracy}")
        print(f"Precision: {test_precision}")
        print(f"Recall: {test_recall}")
        print(f"F1 Score: {test_f1}")

def main():
    file_path = r"C:\Users\aakan\Downloads\heart.csv"  
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = initialize_model()
    train_model(model, X_train, y_train)
    
    y_train_pred = predict(model, X_train)
    y_test_pred = predict(model, X_test)
    
    train_accuracy, train_precision, train_recall, train_f1 = evaluate_metrics(y_train, y_train_pred)
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_metrics(y_test, y_test_pred)
    
    print("Точность на обучающей выборке:", train_accuracy)
    print("Точность на тестовой выборке:", test_accuracy)
    print("\nМетрики на обучающей выборке:")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1}")
    
    print("\nМетрики на тестовой выборке:")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")
    
    cross_validate(X, y, model)

if __name__ == "__main__":
    main()


