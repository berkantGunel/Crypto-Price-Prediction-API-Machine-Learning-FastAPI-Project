# Crypto Price Prediction API

Basit bir yapay zekâ destekli Bitcoin fiyat tahmin uygulaması.  
CoinGecko API'den alınan verilerle Ridge Regression modeli eğitilip, FastAPI üzerinden tahmin servisi olarak sunulur.  

-----

##Özellikler
- CoinGecko API'den son 7 günün verisi alınır.  
- Hareketli ortalama ve volatilite hesaplanır.  
- Ridge Regression modeli ile bir gün sonrası fiyat tahmini yapılır.  
- FastAPI ile '/predict' endpoint'i üzerinden tahmin servisi sunulur.  
- API çağrılarını azaltmak için kısa süreli dosya önbellekleme, ücretsiz api için.

---


## Kurulum
- pip install -r requirements.txt
- .env oluştur:COINGECKO_API_KEY=api_key
- python utils/fetch_data.py
- python utils/preprocess.py
- python train_model.py
- uvicorn api.main:app --reload
- http://127.0.0.1:8000/predict?coin=bitcoin

## Kullanılan Teknolojiler
- Py 3.10
- FastAPI
- sklearn
- pd/np
- requests/dotenv

## Not
Bu proje eğitim amaclıdır.**YTD**
