# Документация API Кредитного Скоринга

## Описание

Модель предсказывает вероятность дефолта (`loan_status = 1`) на основе данных кредитной заявки.

---

## Входные данные

### Числовые признаки

| Признак | Тип | Описание | Ограничения | Статистика |
|---------|-----|----------|-------------|------------|
| `person_income` | float | Годовой доход заявителя ($) | 4,000 – 6,000,000 | mean: 66,075 |
| `person_emp_length` | float | Стаж работы (лет) | 0 – 123 | mean: 4.8 |
| `loan_amnt` | float | Запрашиваемая сумма кредита ($) | 500 – 35,000 | mean: 9,589 |
| `loan_int_rate` | float | Процентная ставка по кредиту (%) | 5.42 – 23.22 | mean: 11.01 |
| `loan_percent_income` | float | Отношение суммы кредита к годовому доходу | 0.0 – 0.83 | mean: 0.17 |

### Категориальные признаки

| Признак | Тип | Описание | Допустимые значения |
|---------|-----|----------|---------------------|
| `person_home_ownership` | str | Тип владения жильём | `RENT` — аренда, `OWN` — собственность, `MORTGAGE` — ипотека, `OTHER` — другое |
| `loan_intent` | str | Цель получения кредита | `PERSONAL` — личные нужды, `EDUCATION` — образование, `MEDICAL` — медицина, `VENTURE` — бизнес, `HOMEIMPROVEMENT` — ремонт, `DEBTCONSOLIDATION` — рефинансирование |
| `loan_grade` | str | Рейтинг кредитоспособности | `A` — высокая, `B` — хорошая, `C` — удовлетворительная, `D` — повышенный риск, `E` — высокий риск, `F` — значительный риск, `G` — наивысший риск |
| `cb_person_default_on_file` | str | Наличие истории дефолтов в кредитном бюро | `Y` — есть дефолты, `N` — нет дефолтов |

**Все поля опциональны.** При отсутствии значения используется дефолт из обучающей выборки.

---

## Пример использования

```python
import joblib
import json
from catboost import CatBoostClassifier

from preprocessor import CreditPreprocessor # обязательно!!!!!!!!!!!!!!!!!!!

# Загрузка
preprocessor = joblib.load('models/preprocessor.pkl')
model = CatBoostClassifier()
model.load_model('models/credit_scoring_model.cbm')

with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

# Заявка (можно передавать частично)
application = {
    'person_income': 55000.0,
    'person_emp_length': 5.0,
    'loan_amnt': 8000.0,
    'loan_int_rate': 11.5,
    'loan_percent_income': 0.15,
    'person_home_ownership': 'RENT',
    'loan_intent': 'PERSONAL',
    'loan_grade': 'B',
    'cb_person_default_on_file': 'N'
}

# Предсказание
X_processed = preprocessor.transform(application)
probability = model.predict_proba(X_processed)[0, 1]
decision = 'REJECT' if probability >= metadata['threshold'] else 'APPROVE'

print(f"Вероятность дефолта: {probability:.3f}")
print(f"Решение: {decision}")
```

---

## Минимальный пример

```python
# Только ключевые поля
minimal = {
    'person_income': 50000.0,
    'loan_amnt': 10000.0,
    'person_home_ownership': 'RENT',
    'cb_person_default_on_file': 'N'
}

result = model.predict_proba(preprocessor.transform(minimal))[0, 1]
```

---

## Метрики модели

| Метрика | Значение |
|---------|----------|
| ROC-AUC | 0.94 |
| Порог классификации | 0.77 |
| F1-score | 0.82 |
| Precision | 0.91 |
| Recall | 0.74 |