# src/preprocess.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging
import os

# Obtém o logger que foi configurado no script principal (train.py)
# Isso garante que todos os logs do projeto sigam o mesmo padrão.
logger = logging.getLogger(__name__)

def feature_eng(df):
    """Aplica engenharia de features no dataframe."""
    logger.info("Iniciando engenharia de features...")
    
    # Converte feriados para formato binário
    df['StateHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0).astype(int) 
    df['SchoolHoliday'] = df['SchoolHoliday'].apply(lambda x: 1 if x != '0' else 0).astype(int) 
    logger.info("Features 'StateHoliday' e 'SchoolHoliday' convertidas para binário.")

    # Cria a média móvel de vendas dos últimos 7 dias por loja
    df['7_days_avg_sales'] = df.groupby(['Store'])['Sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    logger.info("Feature '7_days_avg_sales' (média móvel) criada.")
    
    logger.info("Engenharia de features concluída.")
    return df 

def preprocess_data(df, window_size=7, run_timestamp = 'none'):
    """
    Executa o pipeline completo de pré-processamento: ordena, cria features,
    normaliza e cria sequências para o modelo LSTM.
    """
    logger.info(f"Iniciando pré-processamento com window_size={window_size}.")
    features = ['DayOfWeek', 'Customers', 'Open', 'Promo', '7_days_avg_sales']
    target = 'Sales'

    # Ordena os dados para garantir a consistência das sequências temporais
    logger.info("Ordenando dados por 'Store' e 'Date'...")
    df = df.sort_values(['Store', 'Date'])

    df = df[df['Store'] <= 10]
    # Aplica a engenharia de features
    df = feature_eng(df)
    logger.info(f"Colunas após engenharia de features: {df.columns.tolist()}")

    # Normalização
    logger.info(f"Normalizando as features {features} e o target '{target}' com MinMaxScaler.")
    scaler = MinMaxScaler()
    df[features + [target]] = scaler.fit_transform(df[features + [target]])

    # Criação das sequências
    logger.info("Iniciando criação de sequências para o modelo LSTM...")
    X_list, y_list = [], []
    stores = df['Store'].unique()
    logger.info(f"Encontradas {len(stores)} lojas únicas para processar.")
    
    for store in stores:
        print("Store Analisado: ",store)

        store_data = df[df['Store'] == store]
        for i in range(window_size, len(store_data)):
            X_list.append(store_data[features].iloc[i-window_size:i].values)
            y_list.append(store_data[target].iloc[i])

    logger.info(f"Criação de sequências concluída. Total de {len(X_list)} sequências geradas.")

    X = np.array(X_list)
    y = np.array(y_list)

    # Salva o scaler para uso futuro (ex: em uma API de inferência)
    scaler_path = f'artifacts/scaler_{run_timestamp}.pkl'
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    logger.info(f"Salvando o objeto scaler em '{scaler_path}'...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler salvo com sucesso.")

    return X, y