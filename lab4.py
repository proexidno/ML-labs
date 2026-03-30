import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('Student_Performance.csv')

print(data.isnull().sum())

"""Нет пропущенных значений"""

random_state = 50

label_encoder = LabelEncoder()
data['Extracurricular Activities'] = label_encoder.fit_transform(data['Extracurricular Activities'])

X = data.drop('Performance Index', axis=1)
y = data['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=1/3,
    random_state=random_state
)

print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(random_state=random_state),
    "ElasticNet": ElasticNet(random_state=random_state)
}

trained_models = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    predictions[name] = model.predict(X_test)

metrics = {}

for name, y_pred in predictions.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

    print(f"{name}:")
    print(f"  MSE  = {mse:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  R²   = {r2:.4f}\n")

"""R² хуже у Lasso и ElasticNet"""

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Фактические значения')
    ax.set_ylabel('Предсказанные значения')
    ax.set_title(f'{name}\nR² = {metrics[name]["R²"]:.3f}')
    ax.grid(True)

plt.tight_layout()
plt.show()

feature_names = X.columns

coef_df = pd.DataFrame(index=feature_names)
for name, model in trained_models.items():
    coef_df[name] = model.coef_

intercepts = {name: model.intercept_ for name, model in trained_models.items()}
print("Intercept (свободный член):")
for name, icpt in intercepts.items():
    print(f"  {name}: {icpt:.4f}")

print(coef_df.round(4))

coef_df.T.plot(kind='bar', figsize=(10, 6))
plt.title('Сравнение коэффициентов моделей')
plt.ylabel('Значение коэффициента')
plt.xlabel('Модель')
plt.xticks(rotation=0)
plt.legend(title='Признаки')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""Графики сильно отличаются от линейной регрессии"""

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-4, 1, 50)
l1 = np.linspace(0.1, 0.9, 5)

models_tuned = {
    "Linear Regression": LinearRegression().fit(X_train, y_train),
    "Lasso": LassoCV(alphas=alphas, cv=5, random_state=random_state).fit(X_train, y_train),
    "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=l1, cv=5, random_state=random_state).fit(X_train, y_train)
}

predictions_tuned = {}
for name, model in models_tuned.items():
    predictions_tuned[name] = model.predict(X_test)

cv_results = {}
for name, model in models_tuned.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_results[name] = scores
    print(f"{name}:")
    print(f"  R² (CV): {scores.mean():.4f} ± {scores.std():.4f}\n")

print("Optimal parameter for Lasso")
print(f"alpha: {models_tuned["Lasso"].alpha_:.4f}")
print()

print("Optimal parameters for ElasticNet")
print(f"alpha: {models_tuned["ElasticNet"].alpha_:.4f}")
print(f"l1_cache: {models_tuned["ElasticNet"].l1_ratio_}")

print("• Оптимальные гиперпараметры:")
print(f"  - Lasso: alpha = {models_tuned["Lasso"].alpha_:.6f}")
print(f"  - ElasticNet: alpha = {models_tuned["ElasticNet"].alpha_:.6f}, l1_ratio = {models_tuned["ElasticNet"].l1_ratio_:.2f}")

coef_df = pd.DataFrame(index=feature_names)
for name, model in models_tuned.items():
    coef_df[name] = model.coef_

print(coef_df.round(4))

coef_df.T.plot(kind='bar', figsize=(10, 6))
plt.title('Сравнение коэффициентов моделей')
plt.ylabel('Значение коэффициента')
plt.xlabel('Модель')
plt.xticks(rotation=0)
plt.legend(title='Признаки')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""Графики очень похожи на график линейной регрессии"""

metrics_tuned = {}
for name, y_pred in predictions_tuned.items():
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics_tuned[name] = {'R²': r2, 'RMSE': rmse}
    print(f"  {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_base = model.predict(X_test)
    r2_base = r2_score(y_test, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

    r2_tuned = metrics_tuned[name]['R²']
    rmse_tuned = metrics_tuned[name]['RMSE']

    print(f"\n{name}:")
    print(f"  Базовая:   R² = {r2_base:.4f}, RMSE = {rmse_base:.4f}")
    print(f"  Настроенная: R² = {r2_tuned:.4f}, RMSE = {rmse_tuned:.4f}")
    print(f"  Улучшение R²: {r2_tuned - r2_base:+.4f}")

"""Lasso и ElasticNet увеличит r2 на 2 тысячных, каждый"""
