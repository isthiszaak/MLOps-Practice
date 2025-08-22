# src/evaluate.py

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import tensorflow as tf
import logging

# Obtém o logger configurado no script principal
logger = logging.getLogger(__name__)

def evaluate_model(model_path, X_test, y_test):
    """
    Carrega um modelo treinado, faz predições nos dados de teste
    e calcula as métricas RMSE e MAPE.
    """
    logger.info("="*50)
    logger.info("INICIANDO AVALIAÇÃO DO MODELO")
    logger.info("="*50)
    
    try:
        # Carrega o modelo
        logger.info(f"Carregando modelo do caminho: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Modelo carregado com sucesso.")

        # Faz predições
        logger.info(f"Realizando predições no conjunto de teste com shape: {X_test.shape}")
        y_pred = model.predict(X_test)
        logger.info(f"Predições concluídas. Shape das predições: {y_pred.shape}")

        print("y_teste:", y_test[0])
        print("y_pred:", y_pred[0])


        # Calcula as métricas
        logger.info("Calculando métricas de avaliação: RMSE e MAPE...")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Loga os resultados finais
        logger.info(f"AVALIAÇÃO COMPLETA - RMSE: {rmse:.4f}, MAPE: {mape:.2%}")
        
        return rmse, mape

    except Exception as e:
        logger.error(f"Ocorreu um erro durante a avaliação: {e}", exc_info=True)
        return None, None