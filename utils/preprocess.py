#utils/preprocess.py

import os
import pandas as pd

#zaman serisi
from utils.feature_engineering import(
    FEATURE_COLUMNS,
    ONE_DAY_MINUTES,
    add_time_features,
    minutes_to_steps)

#veri düzeltme ve kayıt
def preprocess_data(input_path="data/raw_data.csv", output_path="data/processed_data.csv"):
    print(f"Veriler yukleniyor:{input_path}")
    df = pd.read_csv(input_path)

    df, step_minutes, horizon_steps = add_time_features(df)
    #zaman temsili özellikleri
    #gelecekteki fiyatı hedef olarak ayarlardik
    
    df["target"] = df["price"].shift(-horizon_steps)#bir gün sonrası fiyat
    df = df.dropna().reset_index(drop=True)#veri temiz.


    #girdiler
    feature_cols = [
        "date",
        *FEATURE_COLUMNS,
        "target",
    ]


    #raw csvyi işleyip processed_data.csv olarak ayarladik
    processed_df = df[feature_cols]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    print(f"islenmiş veri kaydedildi:{output_path} ({len(processed_df)} satir)")
    print(f"kullanilan zaman adimi {step_minutes:.2f} dakika - hedef {horizon_steps} adim ileri git")
    return processed_df


if __name__ == "__main__":
    processed = preprocess_data()
    print(processed.head())
