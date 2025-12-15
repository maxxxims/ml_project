from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import joblib
import os
from pathlib import Path
from .preprocessing import preprocess_text
from sklearn.model_selection import train_test_split
from src.core.config import DIM_TARGET, WEIGHT_FEATURES, DIM_FEATURES, WEIGHT_TARGET


def train_model(
    prepared_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Union[List[str], str],
    model_params: dict,
    val_size: float = 0.2,
    text_features: List[str] = ['title', 'description'],
) -> CatBoostRegressor: 
    _prepared_df = prepared_df.copy()
    _prepared_df = _prepared_df[_prepared_df.groupby('microcat_id')['microcat_id'].transform('count') > 2].reset_index(drop=True)
    
    assert set(feature_cols).intersection(set(_prepared_df.columns)) == set(feature_cols), f"Unknown columns: {set(feature_cols).difference(_prepared_df.columns)}"
    
    train_df, val_df = train_test_split(_prepared_df, test_size=val_size, stratify=_prepared_df['microcat_id'])
    
    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df[target_col],
        text_features=text_features,
        feature_names=list(train_df[feature_cols])
    )
    val_pool = Pool(
        data=val_df[feature_cols],
        label=val_df[target_col],
        text_features=text_features,
        feature_names=list(val_df[feature_cols])
    )
    
    model = CatBoostRegressor(**model_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=100,
    )
    return model
    
    
    
def eval_weight_model(weight_model: CatBoostRegressor, prepared_df: pd.DataFrame) -> dict:
    preds = weight_model.predict(prepared_df[WEIGHT_FEATURES])
    y_true = prepared_df[WEIGHT_TARGET]
    
    metrics = {}
    
    metrics['weight_MSE'] = np.mean((y_true - preds) ** 2)
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
        metrics[f'{dim}_MSE'] = np.mean((y_true - pred) ** 2)
        metrics[f'{dim}_MAE'] = np.mean(np.abs(y_true - pred))
    
    return metrics