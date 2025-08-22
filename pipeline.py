import mlflow
import mlflow.tensorflow

from src.train import train_model

mlflow.set_experiment("sales_forecast_lstm")

with mlflow.start_run():
    mlflow.tensorflow.autolog()  # registra modelo, métricas, hyperparams
    model, history = train_model(X_train, y_train, X_val, y_val)


import FastAPI

app = FastAPI()

