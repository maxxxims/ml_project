import json
from typing import List, Tuple, Union
import pandas as pd
import logging
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from src.core.config import DIM_TARGET, WEIGHT_FEATURES, DIM_FEATURES, WEIGHT_TARGET, DIM_MODEL_PARAMS, WEIGHT_MODEL_PARAMS
from src.core.models import Predictor

logger = logging.getLogger(__name__)


def split_df(df: pd.DataFrame, test_size: float, stratify: bool = True) -> pd.DataFrame:
    if not stratify:
        return train_test_split(df, test_size=test_size)
    _prepared_df = df.copy()
    _prepared_df = _prepared_df[_prepared_df.groupby('microcat_id')['microcat_id'].transform('count') > 2].reset_index(drop=True)
    train_df, val_df = train_test_split(_prepared_df, test_size=test_size, stratify=_prepared_df['microcat_id'])
    return train_df, val_df


def train_one_model(
    prepared_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Union[List[str], str],
    model_params: dict,
    val_size: float = 0.2,
    text_features: List[str] = ['title', 'description'],
) -> CatBoostRegressor: 
   
    assert set(feature_cols).intersection(set(prepared_df.columns)) == set(feature_cols), f"Unknown columns: {set(feature_cols).difference(prepared_df.columns)}"
    
    train_df, val_df = split_df(prepared_df, val_size, stratify=True)
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Val shape: {val_df.shape}")
    
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



def train_models(
    prepared_data: pd.DataFrame,
    weight_max_iterations: int = None,
    dim_max_iterations: int = None,
) -> Tuple[CatBoostRegressor, CatBoostRegressor]:

    logger.info(f"Starting training models stage")
    
    weight_params = WEIGHT_MODEL_PARAMS.copy()
    if isinstance(weight_max_iterations, int):
        weight_params['iterations'] = weight_max_iterations
        
    dim_params = DIM_MODEL_PARAMS.copy()
    if isinstance(dim_max_iterations, int):
        dim_params['iterations'] = dim_max_iterations
    
    weight_model = train_one_model(prepared_data, WEIGHT_FEATURES, WEIGHT_TARGET, weight_params)
    dim_model = train_one_model(prepared_data, DIM_FEATURES, DIM_TARGET, dim_params)

    
    return weight_model, dim_model