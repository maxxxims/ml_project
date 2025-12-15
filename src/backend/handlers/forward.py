from fastapi import APIRouter, Body, File, HTTPException, Depends, Response, UploadFile
import pandas as pd 
import logging

from src.backend.schemas import (
    ForwardRequest,
    ForwardResponse,
    ErrorResponse,
    EvalResponse,
    ForwardBatchResponse
    )
from src.core.models import Predictor
from src.pipline_stages import eval_dim_model, eval_weight_model
from src.core.config import DIM_FEATURES, WEIGHT_FEATURES, DIM_TARGET, WEIGHT_TARGET
from src.backend.utils.utils import validate_file
from src.backend.dependencies import (
    get_predictor,
    PREDICTION_REQUESTS,
    PREDICTION_WEIGHT,
    PREDICTION_HEIGHT,
    PREDICTION_LENGTH,
    PREDICTION_WIDTH
)

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)


@router.post(
    "/forward",
    response_model=ForwardResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def forward(data: ForwardRequest, predictor: Predictor = Depends(get_predictor)):
    """
    Делает предикты из переданных данных.
    Если item_price < 0, то использует среднюю цену по микрокату
    """
    try:
        input_data = data.model_dump()
        if input_data['item_price'] < 0:
            del input_data['item_price']
        
        weight, dimensions = predictor.predict(input_data)
        
        PREDICTION_WEIGHT.observe(weight)
        PREDICTION_HEIGHT.observe(dimensions[0])
        PREDICTION_LENGTH.observe(dimensions[1])
        PREDICTION_WIDTH.observe(dimensions[2])
        
        PREDICTION_REQUESTS.labels(status='success').inc()
        
        response = ForwardResponse(
            weight=weight,
            height=dimensions[0],
            length=dimensions[1],
            width=dimensions[2]
        )
        return response
        
    except ValueError as e:
        PREDICTION_REQUESTS.labels(status='error').inc()
        raise HTTPException(
            status_code=403,
            detail={"error": "Модель не смогла обработать данные", "message": str(e)}
        )
    except Exception as e:
        PREDICTION_REQUESTS.labels(status='error').inc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )
        
        
        
@router.post("/forward_batch",
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def forward_batch_file(
    file: UploadFile = File(...),
    title_column: str = 'title',
    description_column: str = 'description',
    item_price_column: str = 'item_price',
    microcat_id_column: str = 'microcat_id',
    predictor: Predictor = Depends(get_predictor)
):
    """
        Делает предикты из файла. Обрабатывает csv и parquet с переданными колонками. Возвращает csv файл
    """
    df = await validate_file(file, required_columns=[title_column, description_column, item_price_column, microcat_id_column])
    try:
        df.rename(columns={
                'title': title_column,
                'description': description_column,
                'item_price': item_price_column,
                'microcat_id': microcat_id_column
            }, 
            inplace=True
        )   
        logger.info(f"Predicting {len(df)} items")
        
        df_pred = predictor.predict_batch(df[['title', 'description', 'microcat_id', 'item_price']])
        if 'item_id' not in df_pred.columns:
            df_pred['item_id'] = df_pred.index
        
        columns = ['item_id', 'weight_pred', 'height_pred', 'length_pred', 'width_pred']
        
        csv = df_pred[columns].to_csv(index=False)
        return Response(
            content=csv,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=data.csv"}
        )    

    except Exception as e:
        logging.info(e)
        raise HTTPException(
            status_code=403,
            detail={"error": "Модель не смогла обработать данные", "message": str(e)}
        )
        
        
@router.post("/evaluate",
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def eval_batch_csv(
    file: UploadFile = File(...),
    title_column: str = 'title',
    description_column: str = 'description',
    item_price_column: str = 'item_price',
    microcat_id_column: str = 'microcat_id',
    weight_column: str = 'real_weight',
    height_column: str = 'real_height',
    length_column: str = 'real_length',
    width_column: str = 'real_width',
    predictor: Predictor = Depends(get_predictor)
):
    """
        Делает предикты из файла и считает метрики. Обрабатывает csv и parquet с переданными колонками
    """
    required_columns = [title_column, description_column, item_price_column, microcat_id_column,
        weight_column, height_column, length_column, width_column
    ]
    
    df = await validate_file(file, required_columns=required_columns)
        
    try:
        
        df.rename(columns={
                'title': title_column,
                'description': description_column,
                'item_price': item_price_column,
                'microcat_id': microcat_id_column,
                'real_weight': weight_column,
                'real_height': height_column,
                'real_length': length_column,
                'real_width': width_column
            }, 
            inplace=True
        )   
        
        df_features = predictor._prepare_features(df[required_columns])
        metrics = eval_weight_model(predictor.weight_model, prepared_df=df_features)
        metrics.update(
            eval_dim_model(predictor.dim_model, prepared_df=df_features)
        )
        
        metrics = {key: float(value) for key, value in metrics.items()}
           
        return EvalResponse(
            support=len(df_features),
            metrics=metrics
        )

    except Exception as e:
        logging.info(e)
        raise HTTPException(
            status_code=403,
            detail={"error": "Модель не смогла обработать данные", "message": str(e)}
        )