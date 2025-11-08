# utils/fetch_data.py

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


def get_crypto_data(coin_id="bitcoin", days=7, currency="usd"):#btc ve son 7 gün
    load_dotenv()
    #apikey yükleme
    API_KEY = os.getenv("COINGECKO_API_KEY")

    if not API_KEY:
        print("API anahtari yok demo ennd point kullanılacak")

    base_urls = [
        "https://api.coingecko.com/api/v3",
        "https://pro-api.coingecko.com/api/v3",
        #ücretli kullanıyorsanız keyi envden degistirin
    ]
    headers = {}
    if API_KEY:
        headers["x-cg-pro-api-key"] = API_KEY
        base_urls = base_urls[::-1]#pro sürüm yok demoya gectim

    #market verisini cektik
    for base in base_urls:
        url = f"{base}/coins/{coin_id}/market_chart"
        params = {"vs_currency": currency, "days": days}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                break
            elif "10011" in resp.text:
                continue
        except Exception as e:
            print(f"API hatasi: {e}")
    else:
        raise Exception("API istegi basarisiz oldu")

    #raw datamızı olusturduk saniye cinsini okunur tarih yaptık
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["volume"] = [v[1] for v in data.get("total_volumes", [])]
    df["market_cap"] = [m[1] for m in data.get("market_caps", [])]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[["date", "price", "volume", "market_cap"]]


    #csv kayıt
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "raw_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Veri kaydedildi: {out_path} ({len(df)} satir)")
    return df

if __name__ == "__main__":
    try:
        df = get_crypto_data(coin_id="bitcoin", days=7)
        print(df.head())
    except Exception as e:
        print(f"Hata: {e}")
