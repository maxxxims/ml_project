import json
from typing import Tuple
import pandas as pd
import logging

from src.core.models import Predictor


logger = logging.getLogger(__name__)

def extract_data(
    predictor: Predictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"Starting extract_data stage ")
        
    if train_df is not None:
        train_df = predictor._prepare_features(train_df)
        
    if test_df is not None:
        test_df = predictor._prepare_features(test_df)
    
    return train_df, test_df