import numpy as np
import matplotlib.pyplot as plt
from math import e, log
from random import randint

# Данные
x1 = [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06]
y1 = [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
x2 = [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88]
y2 = [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]

# Преобразование данных в матрицы
X1 = np.array(list(zip(x1, y1)))  # Матрица для первого класса
X2 = np.array(list(zip(x2, y2)))    # Матрица для второго класса

# Инициализация весов
weights = [randint(-100, 100) / 100 for _ in range(3)]

# Функция для вычисления z
def weighted_z(point):
    z = np.dot(point, weights[:-1]) + weights[-1]
    return z

# Логистическая функция
def logistic_function(z):
    return 1 / (1 + e ** (-z))

# Функция для вычисления ошибки
def logistic_error():
    errors = []
    inputs = np.vstack((X1, X2))  # Объединяем X1 и X2 для вычисления ошибки
    targets = np.array([0] * len(X1) + [1] * len(X2))  # Целевые значения
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]
        if output == 1:
            output = 0.99999
        if output == 0:
            output = 0.00001
        error = -(target * log(output, e) - (1 - target) * log(1 - output, e))
        errors.append(error)
    return sum(errors) / len(errors)

# Гиперпараметры
lr = 0.1
epochs = 100

# Обучение модели
for epoch in range(epochs):
    inputs = np.vstack((X1, X2))  # Объединяем X1 и X2 для обучения
    targets = np.array([0] * len(X1) + [1] * len(X2))  # Целевые значения
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]
        for j in range(len(weights) - 1):
            weights[j] -= lr * point[j] * (output - target) * (1 / len(inputs))
        weights[-1] -= lr * (output - target) * (1 / len(inputs))
    print(f"Epoch: {epoch}, Error: {logistic_error()}")

# Оценка точности
def accuracy():
    true_outputs = 0
    inputs = np.vstack((X1, X2))  # Объединяем X1 и X2 для оценки точности
    targets = np.array([0] * len(X1) + [1] * len(X2))  # Целевые значения
    for i, point in enumerate(inputs):
        z = weighted_z(point)
        output = logistic_function(z)
        target = targets[i]
        if round(output) == target:
            true_outputs += 1
    return true_outputs, len(inputs)

# Графическая интерпретация
plt.scatter(X1[:, 0], X1[:, 1], color='#ff24cf', label='Класс 0')
plt.scatter(X2[:, 0], X2[:, 1], color='#59e1ff', label='Класс 1')

# Построение границы принятия решения
x_values = np.linspace(0, 9, 100)
y_values = -(weights[0] * x_values + weights[-1]) / weights[1]
plt.plot(x_values, y_values, color='black', label='Граница принятия решения')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Граница принятия решения по логистической регрессии')
plt.legend()
plt.show()

# Вывод точности
true_outputs, total_outputs = accuracy()
print(f"Правильных предсказаний: {true_outputs}")
print(f"Всего предсказаний: {total_outputs}")
print(f"Точность: {true_outputs / total_outputs:.2f}")

# Вывод финальных весов
print("Финальные веса модели:", weights)
