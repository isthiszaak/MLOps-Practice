from fastapi import FastAPI

import pandas as pd
import numpy as np
import tensorflow as tf

from src.preprocess import preprocess

app = FastAPI(title = "API Previsão de Vendas")


# Carrega modelo já treinado
model = tf.keras.models.load_model("artifacts/best_model.h5")

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
    X_input = preprocess(df_input)

    # Previsão
    y_pred = model.predict(X_input)

    return {"predicted_sales": float(y_pred[0,0])}
