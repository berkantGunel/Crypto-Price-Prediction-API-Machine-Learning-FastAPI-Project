from datetime import datetime
import pickle
from pathlib import Path
import sys
import time
import os
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from utils.feature_engineering import FEATURE_COLUMNS, add_time_features


#utils erişim
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


#api okundu pickles yolu tanımlandı
load_dotenv()
API_KEY = os.getenv("COINGECKO_API_KEY")
MODEL_PATH = "./model/model.pkl"
SCALER_PATH = "./model/scaler.pkl"
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

app = FastAPI(title='Crypto Price Prediction API', version="1.0")



def fetch_recent_data(coin_id="bitcoin", days=3, currency="usd"):
    
    os.makedirs("./data", exist_ok=True)
    cache_path = f"./data/cache_{coin_id}.csv"
    cache_ttl = 300
    #cache kontrol, 5dakikadan eskiyse apiye gidiyoruz
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < cache_ttl:
            df = pd.read_csv(cache_path)
            print(f"Cache'den yüklendi: {coin_id}")
            return df

    base_urls = [
        "https://api.coingecko.com/api/v3",
        "https://pro-api.coingecko.com/api/v3",
    ]
    headers = {}
    if API_KEY:
        headers["x-cg-pro-api-key"] = API_KEY
        base_urls = base_urls[::-1]

    for base in base_urls:
        url = f"{base}/coins/{coin_id}/market_chart"
        params = {"vs_currency": currency, "days": days}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
                df["volume"] = [v[1] for v in data.get("total_volumes", [])]
                df["market_cap"] = [m[1] for m in data.get("market_caps", [])]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                df.to_csv(cache_path, index=False)
                print(f"Veri çekildi ve cache'lendi: {coin_id}")
                return df
            elif response.status_code == 429:
                print("Limit aşıldı bekleniyor.")
                time.sleep(5)
                continue
        except Exception as exc:
            print(f"API hatası:{exc}")
    raise Exception("İstek başarısız")




def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat, _, _ = add_time_features(df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    if df_feat.empty:
        raise ValueError("Daha uzun bir pencere cek.")
    return df_feat.tail(1)


@app.get("/predict")
def predict_price(coin: str = Query("bitcoin", description="CoinGecko kimliği")):
    try:
        df = fetch_recent_data(coin_id=coin, days=7)
        df_feat = create_features(df)

        X = df_feat[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X)
        y_pred_log = model.predict(X_scaled)
        y_pred = np.expm1(y_pred_log)

        last_price = df_feat["price"].values[0]
        change = ((y_pred[0] - last_price) / last_price) * 100

        return {
            "coin": coin,
            "current_price": float(last_price),
            "predicted_price": float(y_pred[0]),
            "expected_change_pct": float(change),
            "horizon_minutes": 24 * 60,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/")
def root():
    return {
        "message": "Crypto Price Prediction API çalışıyor. /predict?coin=bitcoin şeklinde sorgulayın."
    }
