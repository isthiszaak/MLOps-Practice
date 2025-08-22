from fastapi import FastAPI

import pandas as pd
import numpy as np
import tensorflow as tf

from src.preprocess import preprocess_input

app = FastAPI(title = "API Previsão de Vendas")


# Carrega modelo já treinado
model = tf.keras.models.load_model("artifacts/best_model_20250822_122935.h5")

@app.post("/predict")
def predict (data: dict):

    """
    Recebe JSON com features para previsão

    Exemplo:
    {
        "Store": 1,
        "Customers": 500,
        "Open": 1,
        "Promo": 1,
        "7_days_avg_sales": 5300,
        "DayOfWeek": 5
    }
    """

    # Converte JSON em dataframe
    df_input = pd.DataFrame([data])
    
    # Pré Processamento
    scaler_path = 'artifacts/scaler_20250822_122935.pkl'
    window_size = 7

    X_input, scaler = preprocess_input(df_input, window_size= window_size, scaler_path = scaler_path)

    # Previsão
    y_pred_scaled = model.predict(X_input)


    dummy = np.zeros((1, len(scaler.scale_)))
    dummy[0, -1] = y_pred_scaled  # coloca a previsão na posição do Sales
    y_pred = scaler.inverse_transform(dummy)[:, -1]  # pega apenas a coluna Sales desnormalizada
    
    return {"predicted_sales": float(y_pred[0])}
