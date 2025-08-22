# src/train.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import preprocess_data
import pandas as pd
import logging
import os
from datetime import datetime

from evaluate import evaluate_model

# --- Configuração do Logging ---
# Cria um timestamp único para esta execução
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define o nome do arquivo de log com o timestamp
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True) # Garante que o diretório de logs exista
log_filename = os.path.join(log_dir, f"training_{run_timestamp}.log")

# Configura o logger para salvar em arquivo e mostrar no console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler() # Envia logs para o console (terminal)
    ]
)

def build_model(input_shape):
    """Constrói e compila o modelo LSTM."""
    logging.info(f"Construindo modelo com input_shape: {input_shape}")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    logging.info("Modelo compilado com otimizador 'adam' e loss 'mse'.")
    return model

def train_model(X_train, y_train, X_val, y_val, model_path):
    """Treina o modelo com os dados fornecidos."""
    logging.info("Configurando callbacks: EarlyStopping e ModelCheckpoint.")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    logging.info("Arquitetura do modelo:")
    model.summary(print_fn=logging.info) # Envia o sumário para o logger

    logging.info("Iniciando o treinamento do modelo...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1, # Mantido em 1 para exemplo rápido, ajuste conforme necessário
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    logging.info("Treinamento concluído.")
    return model, history

if __name__ == "__main__":
    logging.info("="*50)
    logging.info(f"INICIANDO SCRIPT DE TREINAMENTO - RUN ID: {run_timestamp}")
    logging.info("="*50)

    try:
        # Carrega dados
        logging.info("Carregando dados de 'data/train.csv'...")
        df = pd.read_csv('data/train.csv')
        logging.info(f"Dados carregados com sucesso. Shape: {df.shape}")

        # Pré-processamento
        logging.info("Iniciando pré-processamento dos dados com janela de 7 dias...")
        X, y = preprocess_data(df, window_size=7, run_timestamp = run_timestamp)
        logging.info(f"Pré-processamento concluído. Shape de X: {X.shape}, Shape de y: {y.shape}")

        # Split treino/validação simples
        logging.info("Dividindo os dados em conjuntos de treino e validação (80/20)...")
        split = int(0.7 * len(X))
        split_test = int(0.85 * len(X))
        X_train, X_val, X_test = X[:split], X[split:split_test], X[split_test:]
        y_train, y_val, y_test = y[:split], y[split:split_test], y[split_test:]

        logging.info(f"Divisão concluída. Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

        # Define o caminho para salvar o melhor modelo
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True) # Garante que o diretório exista
        model_path = os.path.join(artifacts_dir, f'best_model_{run_timestamp}.h5')
        logging.info(f"O melhor modelo será salvo em: {model_path}")

        model, history = train_model(X_train, y_train, X_val, y_val, model_path=model_path)
        
        # Log do resultado final
        best_val_loss = min(history.history['val_loss'])

        logging.info(f"Treino finalizado. Melhor 'val_loss' alcançada: {best_val_loss:.4f}")

        logging.info(f"Iniciando a Avaliação do modelo")


        evaluate_model(f'artifacts/best_model_{run_timestamp}.h5', X_test, y_test)

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a execução: {e}", exc_info=True)
    
    finally:
        logging.info("="*50)
        logging.info("EXECUÇÃO DO SCRIPT FINALIZADA")
        logging.info("="*50)