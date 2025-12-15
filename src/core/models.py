import logging
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import json
import datetime
import mlflow

from .preprocessing import preprocess_text
from .config import WEIGHT_FEATURES, DIM_FEATURES, mlflow_client

logger = logging.getLogger(__name__)

class Predictor:
    weight_model: CatBoostRegressor
    dim_model: CatBoostRegressor
    metadata: dict
    
    """Класс для предсказания веса и габаритов"""
    
    def __init__(self, microcat_json_path: str = None):
        
        if microcat_json_path is not None:
            with open(microcat_json_path, "r") as f:
                microcat_json = {int(key): value for key, value in json.load(f).items()}
                self.microcat_2_price = {key: value['mean_item_price'] for key, value in microcat_json.items()}
                
            self.microcats = set(microcat_json.keys())
            self.microcat_df = pd.DataFrame().from_dict(microcat_json, orient='index').drop(columns=['mean_item_price'])
        
        self.weight_features = WEIGHT_FEATURES.copy()
        self.dim_features = DIM_FEATURES.copy()
        self.metadata = {
            "weight_model": {},
            "dim_model": {}
        }
        
    
    def load_models_from_files(self, weight_model_path: str, dimensions_model_path: str):
        self.weight_model = self._load_model(weight_model_path)
        self.dim_model = self._load_model(dimensions_model_path)
        
        self.metadata['weight_model'] = {
            'source': 'local_file',
            'load_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'path': weight_model_path,
        }
        
        self.metadata['dim_model'] = {
            'source': 'local_file',
            'load_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'path': dimensions_model_path,
        }
        
        return self
    
    
    def load_models_from_mlflow(self, 
                                weight_model_uri: str,
                                dim_model_uri: str,
                                run_id: str,
                                exp_id: int
                                ):
        weight_model = mlflow.catboost.load_model(weight_model_uri)
        dim_model = mlflow.catboost.load_model(dim_model_uri)
        self.weight_model = weight_model
        self.dim_model = dim_model
        self.metadata['weight_model'] = {
            'source': 'mlflow',
            'load_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'path': weight_model_uri,
            'exp_id': exp_id,
            'run_id': run_id,
        }
        
        self.metadata['dim_model'] = {
            'source': 'mlflow',
            'load_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'path': dim_model_uri,
            'exp_id': exp_id,
            'run_id': run_id,
        }
        return self
            
    
    def _load_model(self, path: str):
        """Загрузка CatBoost модели"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return CatBoostRegressor().load_model(path)
    
    def _get_microcat_name(self, microcat_id: int) -> str:
        return self.microcat_df.loc[microcat_id]['path']
    
    
    def _convert_input_data(self, data: dict) -> pd.DataFrame:    
        input_data = {
            'title': [data['title']],
            'description': [data['description']],
            'microcat_id': [data['microcat_id']]
        }
        if 'item_price' in data:
            input_data['item_price'] = [data['item_price']]
        
        return pd.DataFrame(input_data)
        
    
    def _prepare_features(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка фич из входных данных"""
        microcats = set(input_data['microcat_id'])
        if len(microcats.intersection(self.microcats)) != len(microcats):
            raise ValueError(f"Unknown microcat_id!")    
        
        # текст уже обработан
        # preprocess_text(input_data, col='title')
        # preprocess_text(input_data, col='description')
        
        if 'item_price' not in input_data.columns:
            print("REPLACE MICROCATS WITH PRICES")
            input_data['item_price'] = input_data['microcat_id'].map(self.microcat_2_price)
            input_data['item_price'] = input_data['item_price'].fillna(0)
        
        return input_data.merge(self.microcat_df, left_on='microcat_id', right_index=True, how='inner')



    def predict(self, data: dict) -> pd.DataFrame:
        try:
            input_data = self._convert_input_data(data)
            features = self._prepare_features(input_data)
            
            weight_pred = self.weight_model.predict(features[self.weight_features])[0]
            
            print(self.weight_model.feature_names_, self.weight_model.feature_importances_)
            print(self.dim_model.feature_names_,)
            print(self.weight_features)
            
            dims_pred = self.dim_model.predict(features[self.dim_features])[0]
            
            dimensions = [float(dims_pred[0]), float(dims_pred[1]), float(dims_pred[2])]
            print(weight_pred, dimensions)
            
            return float(weight_pred), dimensions
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._prepare_features(df)

        if len(features) < len(df):
            raise ValueError(f"Unknown microcat_id!")
        
        logger.info(f"Predicting {len(features)} items")
        
        logger.info(f"Predicting weights!")
        weight_pred = self.weight_model.predict(features[self.weight_features])
            
        logger.info(f"Predicting dims!")
        dims_pred = self.dim_model.predict(features[self.dim_features])
        
        height_pred = dims_pred[:, 0]
        length_pred = dims_pred[:, 1]
        width_pred = dims_pred[:, 2]
        
        df['weight_pred'] = weight_pred
        df['height_pred'] = height_pred
        df['length_pred'] = length_pred
        df['width_pred'] = width_pred
        
        return df