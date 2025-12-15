from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from prometheus_client import Counter, Histogram, Gauge
from datetime import datetime

from src.core.models import Predictor
from src.core import config


TABLE_NAME = 'items'
DB_URL = f"postgresql://ml_user:ml_password@cold_feature_storage:5432/cold_features"
engine = create_engine(DB_URL)


PREDICTION_WEIGHT = Histogram(
    'ml_api_predicted_weight_kg',
    'Predicted weight in kilograms',
    buckets=[0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
)

PREDICTION_HEIGHT = Histogram(
    'ml_api_predicted_height_cm',
    'Predicted height in centimeters',
    buckets=[0.1, 1, 5, 10, 20, 30, 50, 100, 200]
)

PREDICTION_LENGTH = Histogram(
    'ml_api_predicted_length_cm',
    'Predicted length in centimeters',
    buckets=[0.1, 1, 5, 10, 20, 30, 50, 100, 200]
)

PREDICTION_WIDTH = Histogram(
    'ml_api_predicted_width_cm',
    'Predicted width in centimeters',
    buckets=[0.1, 1, 5, 10, 20, 30, 50, 100, 200]
)

PREDICTION_REQUESTS = Counter(
    'ml_api_prediction_requests_total',
    'Total prediction requests',
    ['status']
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    predictor = Predictor(microcat_json_path=config.JSON_MICROCAT_PATH)
    predictor.load_models_from_files(
        weight_model_path=config.WEIGHT_MODEL_PATH,
        dimensions_model_path=config.DIMENSIONS_MODEL_PATH
    )
    return predictor