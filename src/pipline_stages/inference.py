import json
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import logging
from catboost import CatBoostRegressor

from src.core.config import DIM_TARGET, WEIGHT_FEATURES, DIM_FEATURES, WEIGHT_TARGET, DIM_MODEL_PARAMS, WEIGHT_MODEL_PARAMS

logger = logging.getLogger(__name__)


def eval_weight_model(weight_model: CatBoostRegressor, prepared_df: pd.DataFrame) -> dict:
    preds = weight_model.predict(prepared_df[WEIGHT_FEATURES])
    y_true = prepared_df[WEIGHT_TARGET]
    
    metrics = {}
    
    metrics['weight_RMSE'] = np.sqrt(np.mean((y_true - preds) ** 2))
    metrics['weight_MAE'] = np.mean(np.abs(y_true - preds))
    
    return metrics


def eval_dim_model(dim_model: CatBoostRegressor, prepared_df: pd.DataFrame) -> dict:
    preds = dim_model.predict(prepared_df[DIM_FEATURES])
    preds_dict = {
        'height': preds[:, 0],
        'length': preds[:, 1],
        'width':  preds[:, 2],
    }

    metrics = {}
    
    for dim, pred in preds_dict.items():
        y_true = prepared_df[f"real_{dim}"]
        metrics[f'{dim}_RMSE'] = np.sqrt(np.mean((y_true - pred) ** 2))
        metrics[f'{dim}_MAE'] = np.mean(np.abs(y_true - pred))
    
    return metrics

def inference_models(
    test_data: pd.DataFrame,
    weight_model: CatBoostRegressor,
    dim_model: CatBoostRegressor,
    
) -> dict:
    logger.info(f"Starting training models stage")
    metrics = eval_weight_model(weight_model, test_data)
    metrics_2 = eval_dim_model(dim_model, test_data)
    metrics.update(metrics_2)
    return metrics