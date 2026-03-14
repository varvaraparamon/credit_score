import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier

from preprocessor import CreditPreprocessor

import warnings
warnings.filterwarnings('ignore')



# ==========================================
#  MAIN PIPELINE
# ==========================================
def main():
    print("Загрузка данных...")
    data = pd.read_csv('../data/credit_risk_dataset.csv')
    print(f"Исходный размер: {data.shape}")
    

    data = data.drop_duplicates()
    data = data.drop(columns=['person_age', 'cb_person_cred_hist_length'], errors='ignore')

    print("Разделение данных...")
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=21
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=21
    )
    
    print("Обучение препроцессора...")
    preprocessor = CreditPreprocessor()
    preprocessor.fit(X_train)
    
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    
    print(f"Размер после обработки: {X_train_proc.shape}")
    

    print("Обучение CatBoost...")
    cat_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    
    best_params = {
        'iterations': 431, 
        'depth': 5, 
        'learning_rate': 0.10763965306425233, 
        'l2_leaf_reg': 2.116248801552587, 
        'random_strength': 0.0011349641784642202, 
        'bagging_temperature': 0.310611777141209, 
        'border_count': 221, 
        'scale_pos_weight': 8.24709157220298
        }

    model = CatBoostClassifier(
        **best_params,
        cat_features=cat_features,
        random_seed=21,
        verbose=100,
        early_stopping_rounds=50,
        task_type='GPU',  
        devices='0'
    )
    
    model.fit(X_train_proc, y_train, eval_set=(X_val_proc, y_val))
    
    print("Подбор оптимального порога...")
    y_proba_val = model.predict_proba(X_val_proc)[:, 1]
    
    best_f1 = 0
    best_threshold = 0.5
    for t in np.linspace(0.1, 0.9, 50):
        y_pred = (y_proba_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    

    y_pred_val = (y_proba_val >= best_threshold).astype(int)
    metrics = {
        'threshold': float(best_threshold),
        'roc_auc': float(roc_auc_score(y_val, y_proba_val)),
        'f1': float(best_f1),
        'precision': float(precision_score(y_val, y_pred_val)),
        'recall': float(recall_score(y_val, y_pred_val))
    }
    
    print(f"Лучший порог: {best_threshold:.3f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   F1: {metrics['f1']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    # ==========================================
    #  СЕРИАЛИЗАЦИЯ
    # ==========================================
    print("Сохранение модели...")
    

    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    
    model.save_model('../models/credit_scoring_model.cbm')
    
    metadata = {
        **metrics,
        'cat_features': cat_features,
        'numeric_features': preprocessor.numeric_features,
        'all_features': list(X_train_proc.columns),
        'log_features': preprocessor.log_features
    }
    
    with open('../models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Сохранено в папку models/:")
    print("   - preprocessor.pkl")
    print("   - credit_scoring_model.cbm")
    print("   - metadata.json")
    
    # ==========================================
    # ПРОВЕРКА ИНФЕРЕНСА
    # ==========================================
    print("\nТестирование инференса...")
    

    prep = joblib.load('../models/preprocessor.pkl')
    mdl = CatBoostClassifier()
    mdl.load_model('../models/credit_scoring_model.cbm')
    
    with open('../models/metadata.json', 'r') as f:
        meta = json.load(f)
    
    sample = pd.DataFrame([{
        'person_income': 50000.0,
        'person_emp_length': np.nan, 
        'loan_amnt': 10000.0,
        'loan_int_rate': 11.5,
        'loan_percent_income': 0.2,
        'person_home_ownership': 'RENT',
        'loan_intent': 'PERSONAL',
        'loan_grade': 'B',
        'cb_person_default_on_file': 'N'
    }])
    

    X_ready = prep.transform(sample)
    proba = mdl.predict_proba(X_ready)[0, 1]
    decision = 'REJECT' if proba >= meta['threshold'] else 'APPROVE'
    
    print(f"   Вероятность дефолта: {proba:.3f}")
    print(f"   Порог: {meta['threshold']:.3f}")
    print(f"   Решение: {decision}")
    print("\n Готово!")

if __name__ == "__main__":
    main()