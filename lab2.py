"""
1. Загрузите данные в DataFrame.
"""

import pandas as pd
student_data = pd.read_csv('ncr_ride_bookings.csv')

"""Выведите первые 5-10 строк."""

print(student_data.head(10))

"""Используйте методы .info(), .describe(), .shape для получения общей информации."""

print(student_data.describe())
print(student_data.info())
print(student_data.shape)

"""Анализ пропусков:
Посчитайте количество и долю пропусков в каждом столбце.
"""

missing_count = student_data.isnull().sum()
missing_percent = (student_data.isnull().mean() * 100)

missing_df = pd.DataFrame({
    'Количество пропусков': missing_count,
    'missing_percent': missing_percent
})

print(missing_df)

"""o Визуализируйте матрицу пропусков с помощью sns.heatmap().o Визуализируйте матрицу пропусков с помощью sns.heatmap()."""

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(student_data.isnull(), cbar=False, cmap='viridis')
plt.title('Матрица пропусков')
plt.xlabel('Столбцы')
plt.ylabel('Объекты')
plt.show()

"""Вывод: Определите столбцы с наибольшим процентом пропусков. Предложите
стратегию их обработки (удаление, заполнение медианой/модой).
"""

top_missing = missing_df.sort_values('missing_percent', ascending=False)

print("Столбцы с наибольшей долей пропусков:")
print(top_missing.head(5))

top_cols = top_missing.head(5).index.tolist()
print("\nИмена столбцов:", top_cols)

"""Анализ числовых признаков:
 Для всех числовых столбцов постройте гистограммы и boxplot'ы.
"""

import numpy as np
numeric_cols = student_data.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    data = student_data[col].dropna()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, color='skyblue')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=data, color='lightgreen')
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()

"""Рассчитайте стандартные статистики (среднее, медиана, стандартное отклонение,
асимметрия) для ключевых числовых признаков.
"""

num_cols = student_data.select_dtypes(include=np.number).columns

stats = student_data.select_dtypes('number').agg(['mean', 'median', 'std', 'skew']).T

print(stats)

"""4. Анализ категориальных признаков:


"""

categorical_cols = student_data.sample(n=100, random_state=42)

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=categorical_cols, x=col, hue=col, palette="Set2", legend=False)
    plt.title(f'Распределение (по сэмплированным данным) в {col}')
    plt.xticks(rotation=45)
    plt.show()

"""o Посчитайте количество уникальных категорий в каждом признаке."""

categorical_cols = student_data.select_dtypes(include=["object", "category"]).columns
unique_counts = student_data[categorical_cols].nunique()
print(unique_counts)

"""Определите категориальные признаки с большим количеством уникальных
значений (высокая кардинальность).
"""

high_cardinality = unique_counts[unique_counts > 20]

print("\nКатегориальные признаки с высокой кардинальностью:\n", high_cardinality)

"""Постройте матрицу корреляций для числовых признаков и визуализируйте ее
тепловой картой (sns.heatmap())
"""

corr_matrix = student_data[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Корреляционная матрица числовых признаков", fontsize=14)
plt.show()

"""Для пар ключевых признаков постройте диаграммы рассеяния (sns.scatterplot())."""

sns.scatterplot(data=student_data, x="Ride Distance", y="Booking Value", hue="Payment Method")
plt.show()

sns.scatterplot(data=student_data, x="Ride Distance", y="Driver Ratings", hue="Vehicle Type")
plt.show()

"""Исследуйте взаимосвязь категориальных и числовых признаков с
помощью boxplot'ов (например, sns.boxplot(x='категория', y='число')).
"""

categorical_cols = [
    "Booking Status", "Vehicle Type", "Pickup Location",
    "Drop Location", "Payment Method"
]

numeric_cols = [
    "Booking Value", "Ride Distance",
    "Driver Ratings", "Customer Rating"
]

for cat in categorical_cols:
    for num in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=student_data, x=cat, y=num, palette="Set2",hue=cat)
        plt.title(f"Boxplot: {num} по категориям {cat}")
        plt.xticks(rotation=45)
        plt.show()

"""Приведите названия столбцов к удобному формату (например, нижний регистр)."""

student_data.columns = (
    student_data.columns.str.strip()
             .str.lower()
             .str.replace(" ", "_")
             .str.replace("-", "_")
)

"""Обработайте пропуски в соответствии с выводами из п.2."""

missing_percent = student_data.isnull().mean() * 100

cols_to_drop = missing_percent[missing_percent > 50].index
student_data = student_data.drop(columns=cols_to_drop)
print("Удалены столбцы:", cols_to_drop.tolist())

num_cols = student_data.select_dtypes(include=['number']).columns
for col in num_cols:
    if student_data[col].isnull().sum() > 0:
        student_data[col] = student_data[col].fillna(student_data[col].median())

cat_cols = student_data.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    if student_data[col].isnull().sum() > 0:
        student_data[col] = student_data[col].fillna(student_data[col].mode()[0])

"""o Преобразуйте категориальные признаки в числовой формат выбранным методом."""

sampled_data = student_data.sample(n=1000, random_state=42)
student_data = pd.get_dummies(sampled_data, columns=cat_cols, drop_first=True)

"""o Выберите один числовой признак с сильными выбросами."""

num_cols = student_data.select_dtypes(include=['number']).columns

outlier_counts = {}
for col in num_cols:
    Q1 = student_data[col].quantile(0.25)
    Q3 = student_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = student_data[(student_data[col] < lower_bound) | (student_data[col] > upper_bound)]
    outlier_counts[col] = len(outliers)

col_with_most_outliers = max(outlier_counts, key=outlier_counts.get)
print("Признак с наибольшим количеством выбросов:", col_with_most_outliers)
print("Количество выбросов:", outlier_counts[col_with_most_outliers])

"""Примените к нему один из методов обработки выбросов (например,
логарифмирование или "обрезку" на основе IQR).
"""

student_data_raw = student_data.copy()
target_col = col_with_most_outliers

Q1 = student_data[target_col].quantile(0.25)
Q3 = student_data[target_col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

student_data[target_col] = np.where(
    student_data[target_col] < lower_bound, lower_bound,
    np.where(student_data[target_col] > upper_bound, upper_bound, student_data[target_col])
)

print(f"Применена IQR-обрезка выбросов к признаку: {target_col}")

"""o Постройте boxplot до и после обработки и прокомментируйте результат."""

target_col = col_with_most_outliers

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=student_data[target_col], ax=axes[0], color="skyblue")
axes[0].set_title(f"{target_col} - после обработки")

sns.boxplot(y=student_data_raw[target_col], ax=axes[1], color="salmon")
axes[1].set_title(f"{target_col} - до обработки")

plt.tight_layout()
plt.show()
