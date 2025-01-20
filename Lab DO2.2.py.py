#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работа №1

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def descriptive_statistics(data):
    desc_stats = data.describe()
    print("Описательная статистика:\n", desc_stats)

def calculate_below_average_percentage(data):
    mean_score = data['score'].mean()
    below_average = data[data['score'] < mean_score]
    percent_below_average = (len(below_average) / len(data)) * 100
    print(f"Процент учащихся ниже среднего: {percent_below_average:.2f}%")

def calculate_failed_percentage(data, threshold=50):
    failed = data[data['score'] < threshold]
    percent_failed = (len(failed) / len(data)) * 100
    print(f"Процент учащихся, не сдавших экзамен: {percent_failed:.2f}%")
    return failed

def plot_pass_fail_pie(data, failed):
    passed = len(data) - len(failed)
    plt.figure(figsize=(6, 6))
    plt.pie([passed, len(failed)], labels=['Сдавшие', 'Не сдавшие'], autopct='%1.1f%%', colors=['#4CAF50', '#FF5722'])
    plt.title('Распределение сдавших и не сдавших экзамен')
    plt.show()

def plot_score_density(data):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data['score'], shade=True)
    plt.title('Ядерная оценка плотности баллов')
    plt.xlabel('Баллы')
    plt.ylabel('Плотность')
    plt.show()

def plot_grade_distribution(data):
    excellent = data[data['score'] >= 85]
    good = data[(data['score'] >= 70) & (data['score'] < 85)]
    satisfactory = data[(data['score'] >= 50) & (data['score'] < 70)]
    unsatisfactory = data[data['score'] < 50]
    
    labels = ['Отлично', 'Хорошо', 'Удовлетворительно', 'Неудовлетворительно']
    sizes = [len(excellent), len(good), len(satisfactory), len(unsatisfactory)]
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Распределение оценок')
    plt.show()

def calculate_gender_percentage(data):
    gender_counts = data['gender'].value_counts(normalize=True) * 100
    print("Процентное соотношение юношей и девушек:\n", gender_counts)

def count_schools(data):
    school_count = data['school_id'].nunique()
    print(f"Количество школ, участвовавших в экзамене: {school_count}")

def count_task_types(data):
    task_v = [col for col in data.columns if col.startswith('V')]
    task_c = [col for col in data.columns if col.startswith('C')]
    print(f"Количество заданий типа В: {len(task_v)}")
    print(f"Количество заданий типа С: {len(task_c)}")
    return task_v, task_c

def calculate_task_completion(data, task_list, task_type):
    task_completion = {task: data[task].mean() * 100 for task in task_list}
    print(f"Процент выполненных заданий типа {task_type}:", task_completion)

def analyze_school_data(data, school_id, task_v, task_c):
    school_data = data[data['school_id'] == school_id]
    
    task_v_completion = {task: school_data[task].mean() * 100 for task in task_v}
    print(f"Процент выполнения заданий типа В для школы {school_id}:", task_v_completion)
    
    task_c_above_50 = {task: school_data[task].mean() * 100 for task in task_c if school_data[task].mean() * 100 > 50}
    print(f"Задания типа С более 50% для школы {school_id}:", task_c_above_50)
    
    gender_scores = school_data.groupby('gender')['score'].mean()
    print(f"Средний балл юношей и девушек по школе {school_id}:\n", gender_scores)

file_path = r'C:\юпитер ноутбук\WPy64-31050\notebooks\ege_results.csv'
data = load_data(file_path)

descriptive_statistics(data)
calculate_below_average_percentage(data)

failed = calculate_failed_percentage(data)
plot_pass_fail_pie(data, failed)

plot_score_density(data)
plot_grade_distribution(data)

calculate_gender_percentage(data)
count_schools(data)

task_v, task_c = count_task_types(data)
calculate_task_completion(data, task_v, 'В')
calculate_task_completion(data, task_c, 'С')

analyze_school_data(data, 101, task_v, task_c)
analyze_school_data(data, 102, task_v, task_c)


# In[2]:


import pandas as pd
import os

data = {
    "student_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "school_id": [101, 101, 102, 102, 103, 104, 105, 101, 102, 104, 103, 105, 104, 102, 101],
    "gender": ["M", "F", "F", "M", "M", "F", "M", "F", "F", "M", "F", "M", "F", "M", "F"],
    "score": [78, 92, 55, 63, 47, 88, 35, 72, 84, 91, 49, 69, 73, 45, 78],
    "V1": [1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    "V2": [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    "V3": [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    "C1": [0.8, 1, 0.5, 0.6, 0.4, 0.9, 0.2, 0.6, 0.8, 0.9, 0.5, 0.7, 0.8, 0.3, 0.9],
    "C2": [0.7, 0.9, 0.3, 0.5, 0.5, 1, 0.1, 0.7, 0.6, 1, 0.4, 0.6, 0.7, 0.2, 0.8],
    "C3": [0.9, 1, 0.6, 0.7, 0.2, 0.8, 0.3, 0.5, 1, 1, 0.3, 0.5, 0.6, 0.4, 0.9]
}

save_path = os.path.join(os.getcwd(), "ege_results.csv") 

df = pd.DataFrame(data)
df.to_csv(save_path, index=False)
print(f"Файл ege_results.csv успешно создан по пути: {save_path}")

