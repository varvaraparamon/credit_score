import joblib
import json
from catboost import CatBoostClassifier

from preprocessor import CreditPreprocessor # обязательно!!!!!!!!!!!!!!!!!!!

def main():
    prep = joblib.load('../models/preprocessor.pkl')
    mdl = CatBoostClassifier()
    mdl.load_model('../models/credit_scoring_model.cbm')
    
    with open('../models/metadata.json', 'r') as f:
        meta = json.load(f)


    # ======= ПЕРВЫЙ ТЕСТ =========
    
    sample = {
        'person_income': 50000.0,
        'person_emp_length': None, ## можно None пихать вместо любого значения
        'loan_amnt': 10000.0,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.2,
        'person_home_ownership': 'RENT',
        'loan_intent': 'PERSONAL',
        'loan_grade': 'B',
        'cb_person_default_on_file': 'N'
    }
    

    X_ready = prep.transform(sample)
    proba = mdl.predict_proba(X_ready)[0, 1]
    decision = 'REJECT' if proba >= meta['threshold'] else 'APPROVE'
    
    print("=== Пример №1 ===")
    print(f"   Вероятность дефолта: {proba:.3f}")
    print(f"   Порог: {meta['threshold']:.3f}")
    print(f"   Решение: {decision}")



    # ======= ВТОРОЙ ТЕСТ =========
    sample = {
        'person_income': 60000.0,
        'person_emp_length': None,
        'loan_amnt': 15000.0,
        # loan_int_rate                      можно вот например просто даже не указывать
        'loan_percent_income': 0.25,
        'person_home_ownership': 'MORTGAGE',
        'loan_intent': 'HOMEIMPROVEMENT',
        # loan_grade                        тоже отсутствует
        'cb_person_default_on_file': 'Y'
    }
    

    X_ready = prep.transform(sample)
    proba = mdl.predict_proba(X_ready)[0, 1]
    decision = 'REJECT' if proba >= meta['threshold'] else 'APPROVE'
    
    print("\n\n=== Пример №2 ===")
    print(f"   Вероятность дефолта: {proba:.3f}")
    print(f"   Порог: {meta['threshold']:.3f}")
    print(f"   Решение: {decision}")





if __name__ == "__main__":
    main()