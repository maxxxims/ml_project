import os
from pathlib import Path
from dataclasses import dataclass
from mlflow.client import MlflowClient

@dataclass
class MLFlowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "iwgh_prediction"
    registered_model_name: str = "iwgh_model"
    
    @classmethod
    def from_env(cls):
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "iwgh_prediction"),
        )

mlflow_client = MlflowClient(tracking_uri=MLFlowConfig.from_env().tracking_uri)


WEIGHT_MODEL_PATH = "/app/data/models/weight_model.cbm"
DIMENSIONS_MODEL_PATH = "/app/data/models/dim_model.cbm"
JSON_MICROCAT_PATH = "/app/data/microcat.json"


WEIGHT_FEATURES = [
    'title',
    'description',
    'mean_weight',
    'std_weight',
    'item_price',
]
DIM_FEATURES = ['title', 'description', 'item_price', 'mean_weight', 'std_weight', 'mean_height',
             'std_height', 'mean_length', 'std_length', 'mean_width', 'std_width']


WEIGHT_TARGET = 'real_weight'
DIM_TARGET = ['real_height', 'real_length', 'real_width']


WEIGHT_MODEL_PARAMS = {    
    'iterations': 20,
    'learning_rate': 0.15,
    'l2_leaf_reg': 10,
    'depth': 6,
    'random_strength': 6,
    'bagging_temperature': 0,
    'text_processing': {
    'tokenizers':[{
        'tokenizer_id': 'Sense',
        'separator_type': 'BySense',
        'token_types': ['Word', 'Number']
    }],

    "dictionaries" : [{
        "dictionary_id" : "BiGram",
        "max_dictionary_size" : "50000",
        # "occurrence_lower_bound" : "5",
        "gram_order" : "2"
    }, {
        "dictionary_id" : "Word",
        "max_dictionary_size" : "50000",
        # "occurrence_lower_bound" : "5",
        "gram_order" : "1"
    }],

    "feature_processing" : {
        "default" : [{
            "tokenizers_names" : ["Sense"],
            "dictionaries_names" : ["Word", "BiGram"],
            "feature_calcers" : ["BoW"]
        }],
    }
    },
    'boosting_type': 'Plain',
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'early_stopping_rounds': 500,
    'use_best_model': True,
    'task_type': 'CPU',
    'thread_count': 4
}


DIM_MODEL_PARAMS = {    
    'iterations': 20,
    'learning_rate': 0.15,
    'l2_leaf_reg': 10,
    'depth': 6,
    'random_strength': 6,
    'bagging_temperature': 0,
    'text_processing': {
    'tokenizers':[{
        'tokenizer_id': 'Sense',
        'separator_type': 'BySense',
        'token_types': ['Word', 'Number']
    }],

    "dictionaries" : [{
        "dictionary_id" : "BiGram",
        "max_dictionary_size" : "50000",
        # "occurrence_lower_bound" : "5",
        "gram_order" : "2"
    }, {
        "dictionary_id" : "Word",
        "max_dictionary_size" : "50000",
        # "occurrence_lower_bound" : "5",
        "gram_order" : "1"
    }],

    "feature_processing" : {
        "default" : [{
            "tokenizers_names" : ["Sense"],
            "dictionaries_names" : ["Word", "BiGram"],
            "feature_calcers" : ["BoW"]
        }],
    }
    },
    'boosting_type': 'Plain',
    'loss_function': 'MultiRMSE',
    'eval_metric': 'MultiRMSE',
    'early_stopping_rounds': 500,
    'use_best_model': True,
    'task_type': 'CPU',
    'thread_count': 4
}