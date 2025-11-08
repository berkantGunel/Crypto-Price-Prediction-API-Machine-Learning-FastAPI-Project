# feature_engineering.py
import pandas as pd

#gün->dk, hedef tahmin araligi
ONE_DAY_MINUTES = 24 * 60




ROLLING_WINDOWS_MINUTES = {
    "ma_1h": 60,#son bir saat
    "ma_6h": 360,#6 saat
    "ma_24h": 1440,#24saat
}

#fiyat degisimi
VOLATILITY_WINDOWS_MINUTES = {
    "volatility_1h": 60,
    "volatility_6h": 360,
}


def infer_step_minutes(df: pd.DataFrame) -> float:
    #veri iki satırdan azsa varsayılan 5dk kabul ettik
    if len(df) < 2:
        return 5.0
    deltas = df["date"].diff().dropna().dt.total_seconds() / 60
    median_delta = deltas.median()
    return float(median_delta) if pd.notnull(median_delta) else 5.0
    #degilse her satır arasındaki zaman farkını hesaplayıp dakika cinsine cevir 

def minutes_to_steps(minutes: int, step_minutes: float) -> int:
    return max(1, int(round(minutes / step_minutes)))
#mesela bir saatlik pencere 60*5=12




def add_time_features(df: pd.DataFrame):
    df = df.copy()
    if "date" not in df:
        raise ValueError("DataFrame must contain a 'date' column")


    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    step_minutes = infer_step_minutes(df)#veri noktaları arasındaki zaman farkı
    horizon_steps = minutes_to_steps(ONE_DAY_MINUTES, step_minutes)


    #fiyat yüzde değişimi
    df["price_change_pct"] = df["price"].pct_change()*100
    for feature_name,minutes in ROLLING_WINDOWS_MINUTES.items():
        window= minutes_to_steps(minutes, step_minutes)
        df[feature_name] = df["price"].rolling(window=window, min_periods=window).mean()

    #fiyat std kontrolü 
    for feature_name, minutes in VOLATILITY_WINDOWS_MINUTES.items():
        window = max(2, minutes_to_steps(minutes, step_minutes))
        df[feature_name] = df["price"].rolling(window=window, min_periods=window).std()

    return df, step_minutes, horizon_steps


#X
FEATURE_COLUMNS = [
    "price",
    "volume",
    "market_cap",
    "price_change_pct",
    *ROLLING_WINDOWS_MINUTES.keys(),
    *VOLATILITY_WINDOWS_MINUTES.keys(),
]
