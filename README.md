# Отчет к домашней работе
Выполнила: Зинченко Анна, ИСП-221с

### Задание:
1. Подобрать набор x1 y1 и x2 y2. ПРЕДСТАВИТЬ ДАННЫЕ В ВИДЕ ДВУХ МАТРИЦ Х1 и Х2. 
2. Выбрать гиперпараметры lr и количество эпох обучения.
3. Оценить качество полученной модели сделать графическую интерпретацию.
4. Выполнить работу и сравнить удобство Google Colab и Kaggle.
5. Сохранить варианты выполнения работы на GitHub.

1. Подбор набора данных
Были использованы следующие наборы данных:
```
# Данные
x1 = [0.38, 1.42, 0.55, 1.34, 1.76, 1.62, 0.83, 0.84, 1.77, 1.06]
y1 = [1.79, 0.54, 0.34, 0.678, 1.64, 0.92, 1.49, 0.3, 0.7, 0.99]
x2 = [3.9, 6.14, 6.1, 2.11, 3.23, 1.62, 1.88]
y2 = [4.93, 4.95, 0.97, 0.77, 0.43, 4.61, 0.25]

Преобразование данных в матрицы:
X1 = np.array(list(zip(x1, y1)))  # Матрица для первого класса
X2 = np.array(list(zip(x2, y2)))    # Матрица для второго класса
```
2. Гиперпараметры
В качестве гиперпараметров были выбраны:
```
lr = 0.1
epochs = 100
```
3. Оценка качества полученной модели и графическая интерпретация
* Оценка качества полученной модели
```
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

    true_outputs, total_outputs = accuracy()
    print(f"Правильных предсказаний: {true_outputs}")
    print(f"Всего предсказаний: {total_outputs}")
    print(f"Точность: {true_outputs / total_outputs:.2f}")

    print("Финальные веса модели:", weights)
```
* Графическая интерпретация
```
plt.scatter(X1[:, 0], X1[:, 1], color='#ff24cf', label='Класс 0')
plt.scatter(X2[:, 0], X2[:, 1], color='#59e1ff', label='Класс 1')

x_values = np.linspace(0, 9, 100)
y_values = -(weights[0] * x_values + weights[-1]) / weights[1]
plt.plot(x_values, y_values, color='black', label='Граница принятия решения')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Граница принятия решения по логистической регрессии')
plt.legend()
plt.show()
```

Запуск кода: 
<div align="left">
  <img src="https://github.com/domosedochka/dz_razrabotka_pm/blob/main/Снимок%20экрана%202024-12-20%20172010.png" width="1920" />
</div>

4. Сравнение удобства на Google Colab и Kaggle:
* Google Colab
Преимущества включают простой доступ через аккаунт Google, удобную загрузку/выгрузку файлов с Google Диска и GitHub, а также быстрое выполнение кода.
<div align="left">
  <img src="https://github.com/domosedochka/dz_razrabotka_pm/blob/main/Снимок%20экрана%202024-12-20%20170731.png" width="1920" />
</div> 
<div align="left">
  <img src="https://github.com/domosedochka/dz_razrabotka_pm/blob/main/Снимок%20экрана%202024-12-20%20170924.png" width="1920" />
</div>

* Kaggle
Не требует регистрации для создания блокнота. Однако, время выполнения кода было значительно дольше, чем в Google Colab.
<div align="left">
  <img src="https://github.com/domosedochka/dz_razrabotka_pm/blob/main/Снимок%20экрана%202024-12-20%20172128.png" width="1920" />
</div>
<div align="left">
  <img src="https://github.com/domosedochka/dz_razrabotka_pm/blob/main/Снимок%20экрана%202024-12-20%20172142.png" width="1920" />
</div>

## Итог
Оба сервиса предоставляют удобные средства для работы с кодом, но Google Colab показал преимущество в скорости выполнения.
Kaggle, в свою очередь, ориентирован на пользователей, работающих с задачами машинного обучения, и предоставляет мощные инструменты для анализа данных, такие как встроенные наборы данных и готовые к использованию GPU.



