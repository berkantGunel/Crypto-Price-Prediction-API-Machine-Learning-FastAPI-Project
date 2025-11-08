#train_model.py

import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#utilse erişim
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.feature_engineering import FEATURE_COLUMNS  #giriş veri

DATA_PATH = Path("data/processed_data.csv")#islenen
MODEL_DIR = Path('model')#model kayit yer
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


def evaluate_models(X_train, X_test, y_train, y_test):

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=31,
            n_jobs=-1,#düşük sistemse kaldır
            #inceleme icin verbose ekleyebilirsin
        ),
    }
    results = {}
    fitted = {}
    
    
    
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        fitted[name] = model
        print(f"\n{name.upper()} results MAE: {mae:.4f} RMSE {rmse:.4f} R2 {r2:.4f}")
    return results, fitted


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} yok\nOnce preprocess calistir")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if df.empty:
        raise ValueError("Veri cok kısa")

    X = df[FEATURE_COLUMNS].values#featurec yani giris degerler
    y = np.log1p(df["target"].values)#bagimli degisken, 
    #varyans stabile edildi ve dagilimin degiskenligin yaklasık sabit olması ayarlandı 
    
    
    #80 eğitim %20 test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    #standartizasyon işlemleri mean 0 std 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #sonuc ve modeller
    results, fitted_models = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    #hata kare. en düsük olanı en iyi model secip hiperparametreleri ayarladık
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_params = fitted_models[best_name].get_params()
    print(f"\nBest model: {best_name} (RMSE={results[best_name]['RMSE']:.4f})")

    #model tuning
    scaler_full = StandardScaler().fit(X)
    X_full_scaled = scaler_full.transform(X)

    best_model_class = fitted_models[best_name].__class__
    best_model = best_model_class(**best_params)
    best_model.fit(X_full_scaled, y)

    
    #model ve scaleri kaydettik
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler_full, f)


    print(f"\nModel kaydedildi {MODEL_PATH}")
    print(f"Scaler kaydedildi {SCALER_PATH}")
    return results


if __name__ == "__main__":
    metrics = train()
    print("Results")
    for name, vals in metrics.items():
        print(
            f"{name}: MAE={vals['MAE']:.4f} RMSE={vals['RMSE']:.4f} R2={vals['R2']:.4f}"
        )
