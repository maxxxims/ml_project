from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy import text
import pandas as pd
import io
import logging
from datetime import datetime
from mlflow.tracking import MlflowClient
import mlflow

from src.backend.utils.utils import validate_file
from src.backend.schemas import ColdStorageResponse, DeployRequest, RetrainRequest
from src.backend.dependencies import engine, TABLE_NAME, get_predictor
from ml_pipline.pipline import MLPipeline
from src.pipline_stages import split_df
from src.core.config import mlflow_client
from src.core.models import Predictor



router = APIRouter(prefix="", tags=["pipline"])
logger = logging.getLogger(__name__)


@router.get("/db_info", response_model=ColdStorageResponse)
async def get_info():
    """
    Показывает количество строк и столбцов в холодном хранилище
    """
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    return ColdStorageResponse(rows=len(df), columns=list(df.columns))
    

@router.put("/add_data")
async def upload_to_cold_storage(
    file: UploadFile = File(...),
    title_column: str = 'title',
    description_column: str = 'description',
    item_price_column: str = 'item_price',
    microcat_id_column: str = 'microcat_id',
    weight_column: str = 'real_weight',
    height_column: str = 'real_height',
    length_column: str = 'real_length',
    width_column: str = 'real_width',
):
    """
        Загружает данные в холодное хранилище
    """
    
    required_columns = [title_column, description_column, item_price_column, microcat_id_column,
        weight_column, height_column, length_column, width_column
    ]
    
    df = await validate_file(file, required_columns=required_columns)
    df = df[required_columns]
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
        uploaded_at = datetime.now()
        # Загружаем в БД
        df.to_sql(
            TABLE_NAME,
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        return {
            "message": "Данные успешно загружены",
            "filename": file.filename,
            "rows_uploaded": len(df),
            "table": TABLE_NAME,
            "timestamp": uploaded_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Ошибка загрузки: {str(e)}")



@router.put("/retrain")
async def retrain(
    data: RetrainRequest,
    predictor: Predictor = Depends(get_predictor),
):
    """
        Запускает пайплайн переобучения модели.
        max_train_iterations - максимальное количество итераций обучения. Если None или < 0, то используется дефолтное
    """
    
    experiment_name = data.experiment_name
    max_train_iterations = data.max_train_iterations
    
    if max_train_iterations < 1:
        max_train_iterations = None
    
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    
    train_df, test_df = split_df(df=df, test_size=0.2, stratify=True)
    
    logger.info(f"df size = {train_df.shape}")
    
    ml_pipline = MLPipeline(experiment_name=experiment_name)
    
    result = ml_pipline.run_pipeline(
        predictor=predictor,
        train_data=train_df,
        test_data=test_df,
        weight_max_iterations=max_train_iterations,
        dim_max_iterations=max_train_iterations,
    )    
    return result

    
    
    
@router.get("/metrics/{experiment_id}")
async def get_metrics(experiment_id: str):
    """
    Возвращает всю информацию по эксперименту
    """
    try:
        runs = mlflow_client.search_runs(
            experiment_ids=[str(experiment_id)],
            order_by=["start_time DESC"]
        )
        
        if not runs:
            return {"error": f"Experiment {experiment_id} not found or has no runs"}
        
        metrics_data = []
        
        for run in runs:
            run_metrics = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
            metrics_data.append(run_metrics)
        
        return {
            "experiment_id": experiment_id,
            "total_runs": len(runs),
            "latest_run": runs[0].info.run_id if runs else None,
            "runs": metrics_data
        }
    
    except Exception as e:
        return {"error": str(e)}


@router.post("/deploy/{experiment_id}")
async def deploy_experiment(
    experiment_id: str,
    data: DeployRequest,
    predictor: Predictor = Depends(get_predictor)
):
    """
    Заменяет текущую модель на модель из указанного experiment_id и run_id.
    Если run_id пустая строка, то берется последний запуск
    """
    try:
        logger.info(f"Deploying model from experiment {experiment_id} and run = {data.run_id}")
        runs = mlflow_client.search_runs(
            experiment_ids=[str(experiment_id)],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            return {"error": f"Experiment {experiment_id} not found or has no runs"}
        
        if data.run_id == "":
            run = runs[0]
            run_id = run.info.run_id
        else:
            run_id = data.run_id
            if run_id not in [run.info.run_id for run in runs]:
                raise HTTPException(status_code=404, detail="Run not found")
                
        weight_model_uri = f"runs:/{run_id}/model_weight"
        dim_model_uri = f"runs:/{run_id}/model_dim"
        
        predictor.load_models_from_mlflow(
            weight_model_uri=weight_model_uri,
            dim_model_uri=dim_model_uri,
            run_id=run_id,
            exp_id=experiment_id
        )
        
        return {
            "message": f"Model deployed successfully",
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error deploying model: {str(e)}")



@router.delete("/drop_db")
async def drop_db():
    """Удаляет все данные из холодного хранилища"""
    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLE_NAME} CASCADE"))
        conn.commit()
    return {"status": "ok"}
    
